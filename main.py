import timm
import torch
import random
import tempfile

from timm.data import create_dataset, create_loader
from timm.utils import accuracy, AverageMeter


def new_loader(root: str, split: str, batch_size=32, is_training=True):
    dataset = create_dataset(root=root, split=split, name='', is_training=is_training)
    return create_loader(
        dataset,
        input_size=(3, 224, 224),
        batch_size=batch_size,
        interpolation='bicubic',
        num_workers=2,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        crop_pct=0.95,
        crop_mode='center',
        device='cpu',
        is_training=is_training,
        worker_seeding=lambda _: 42,
    )


train_loader = new_loader(root='../data/nhl/', split='train', batch_size=128)  # 261
test_loader = new_loader(root='../data/nhl/', split='test', batch_size=64)  # 84
val_loader = new_loader(root='../data/nhl/', split='val', batch_size=29, is_training=False)

# train_loader = test_loader = val_loader


def load_model():
    return timm.create_model('coatnet_0_rw_224', checkpoint_path='../model/model_best.pth.tar', num_classes=3)


# Evaluate model before pruning
def evaluate_model():
    top1 = AverageMeter()
    top2 = AverageMeter()

    model.eval()
    with torch.no_grad():
        evaluate_loader = new_loader(root='../data/nhl/', split='test', batch_size=84, is_training=False)
        for batch_idx, (input, target) in enumerate(evaluate_loader):
            # compute output
            output = model(input)
            # measure accuracy and record loss
            acc1, acc2 = accuracy(output.detach(), target, topk=(1, 2))
            top1.update(acc1.item(), input.size(0))
            top2.update(acc2.item(), input.size(0))

            print(
                'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                'Acc@2: {top2.val:>7.3f} ({top2.avg:>7.3f})'.format(
                    top1=top1,
                    top2=top2
                )
            )
    return top1.avg, top2.avg


# %%
def extract_features_attention(m, x):
    B, C, H, W = x.shape
    q, k, v = m.qkv(x).view(B, m.num_heads, m.dim_head * 3, -1).chunk(3, dim=2)
    attn_bias = m.rel_pos.get_bias()

    outputs = []
    for head_idx in range(m.num_heads):
        x = torch.nn.functional.scaled_dot_product_attention(
            q[:, head_idx, :, :].transpose(-1, -2).contiguous(),
            k[:, head_idx, :, :].transpose(-1, -2).contiguous(),
            v[:, head_idx, :, :].transpose(-1, -2).contiguous(),
            attn_mask=attn_bias[:, head_idx],
            dropout_p=m.attn_drop.p if m.training else 0.,
        ).transpose(-1, -2).reshape(B, -1, H, W)
        outputs.append(x)
    return torch.cat(outputs, dim=1)


def extract_features_mb_conv_block(m, x):
    result = {}
    # shortcut = m.shortcut(x)
    x = m.pre_norm(x)
    x = m.down(x)

    # 1x1 expansion conv & norm-act
    x = m.conv1_1x1(x)
    # result[m.conv1_1x1] = x
    x = m.norm1(x)

    # depthwise / grouped 3x3 conv w/ SE (or other) channel attention & norm-act
    x = m.conv2_kxk(x)
    result['conv2_kxk'] = x
    # x = m.se_early(x)
    # result[m.se_early] = x
    # x = m.norm2(x)

    # # 1x1 linear projection to output width
    # x = m.conv3_1x1(x)
    # x = m.drop_path(x) + shortcut
    return result


def zero_weights_attention(m, indexes):
    dim_head = m.dim_head
    assert m.num_heads == len(indexes)

    for hi, should_prune in enumerate(indexes):
        if not should_prune:
            continue
        for i in range(3):  # QKV indexes
            part_idx = dim_head * i
            head_idx = hi * dim_head * 3

            s = part_idx + head_idx
            e = s + dim_head
            m.qkv.weight.data[s:e, :] = .0
            m.qkv.bias.data[s:e] = .0

    # Do not update the head number, because it is not removed from the model
    # m.num_heads -= indexes.sum()


def zero_weights_filter(m, indexes):
    if not isinstance(indexes, dict):
        return

    for layer, idxs in indexes.items():
        layer = m.__getattr__(layer)
        for fi, should_prune in enumerate(idxs):
            if not should_prune:
                continue

            if isinstance(layer, timm.layers.squeeze_excite.SEModule):
                layer.fc2.weight.data[fi, :] = .0
                continue

            layer.weight.data[fi, :] = .0
            # m.conv1_1x1.weight.data[fi, :] = .0
            # m.conv2_kxk.weight.data[fi, :] = .0
            # m.se_early.fc2.weight.data[fi, :] = .0

    # Do not update the filter number, because it is not removed from the model


def random_importance(m, ratio, module_outputs, model_outputs, y):
    if isinstance(m, timm.models.maxxvit.Attention2d):
        num_heads = m.num_heads
        indexes = [0] * num_heads
        indexes[:int(num_heads * ratio)] = [1] * int(num_heads * ratio)
        random.shuffle(indexes)
        return indexes


def decision_tree_importance(m, ratio, module_outputs, y, x_test, y_test):
    from sklearn.metrics import f1_score

    features_num = 0
    if isinstance(m, timm.models.maxxvit.MbConvBlock):
        print('mbconv', m.conv1_1x1.weight.shape[0])
        features_num = m.conv1_1x1.weight.shape[0]
        module_outputs = module_outputs['conv2_kxk']
        x_test = x_test['conv2_kxk']
        # features_num = m.conv1_1x1.weight.shape[0] + m.conv2_kxk.weight.shape[0] + m.se_early.fc2.weight.shape[0]
        # module_outputs = torch.cat([module_outputs[m.conv1_1x1], module_outputs[m.conv2_kxk], module_outputs[m.se_early]], dim=1)
    if isinstance(m, timm.models.maxxvit.Attention2d):
        print('attention', m.num_heads)
        features_num = m.num_heads

    print(ratio)
    if ratio <= 0:
        return [0] * features_num

    filter_f1_scores = []
    print('start decision tree')
    for i in range(features_num):
        filter_output = module_outputs[:, i, :, :].reshape(len(y), -1).detach()
        clf.fit(filter_output, y)

        test_output = x_test[:, i, :, :].reshape(len(y_test), -1).detach()
        pred = clf.predict(test_output)
        score = f1_score(y_test, pred, average='macro')
        filter_f1_scores.append(score)

    print('decision tree importance')
    print(filter_f1_scores)
    pivot = sorted(filter_f1_scores)[int(features_num * ratio)]
    print('pivot: ', pivot)
    indexes = [1 if score <= pivot else 0 for score in filter_f1_scores]

    # fix for under pruning
    if sum(indexes) < ratio * features_num:
        # Flip some 0s to 1s
        position_with_zeros = [i for i, x in enumerate(indexes) if x == 0]
        random.shuffle(position_with_zeros)
        for i in range(int(ratio * features_num - len(position_with_zeros))):
            indexes[position_with_zeros[i]] = 1

    # Fix for over pruning
    if sum(indexes) > ratio * features_num:
        # Flip some 1s to 0s
        position_with_ones = [i for i, x in enumerate(indexes) if x == 1]
        random.shuffle(position_with_ones)
        for i in range(int(len(position_with_ones) - (ratio * features_num))):
            indexes[position_with_ones[i]] = 0

    print(indexes)
    print('prune ratio:', sum(indexes) / len(indexes))

    if isinstance(m, timm.models.maxxvit.MbConvBlock):
        return {
            'conv2_kxk': indexes
        }
    #     shapes = [m.conv1_1x1.weight.shape[0], m.conv2_kxk.weight.shape[0], m.se_early.fc2.weight.shape[0]]
    #     ind = {
    #         m.conv1_1x1: indexes[:shapes[0]],
    #         m.conv2_kxk: indexes[shapes[0]: shapes[0] + shapes[1]],
    #         m.se_early: indexes[shapes[0] + shapes[1]:]
    #     }
    #     return ind

    return indexes


def store_filter_output(self, x):
    filter_output = extract_features_mb_conv_block(self, x)

    temp = tempfile.NamedTemporaryFile(prefix='maxxvit_mbconv_', suffix='.pth')
    torch.save(filter_output, temp.name)
    self.filter_output = temp

    return self.old_forward(x)


def store_head_output(self, x, shared_rel_pos=None):
    attention_output = extract_features_attention(self, x)

    temp = tempfile.NamedTemporaryFile(prefix='maxxvit_attention_', suffix='.pth')
    torch.save(attention_output, temp.name)
    self.attention_output = temp

    return self.old_forward(x, shared_rel_pos)


def prune(conv_ratio, trans_ratio, eval_function):
    # evaluate_model()
    # print('eval complete')

    # Alter function for store intermediate outputs:
    for m in model.modules():
        if isinstance(m, timm.models.maxxvit.MbConvBlock):
            m.old_forward = m.forward.__get__(m, type(m))
            m.forward = store_filter_output.__get__(m, type(m))

        if isinstance(m, timm.models.maxxvit.Attention2d):
            m.old_forward = m.forward.__get__(m, type(m))
            m.forward = store_head_output.__get__(m, type(m))

    # Forward to get data
    inputs, labels = next(iter(train_loader))
    model(inputs)

    train_data = {}
    for m in model.modules():
        if isinstance(m, timm.models.maxxvit.MbConvBlock):
            train_data[m] = m.filter_output

        if isinstance(m, timm.models.maxxvit.Attention2d):
            train_data[m] = m.attention_output

    print('temp Train data')

    # Forward to get data
    test_inputs, test_labels = next(iter(test_loader))
    model(test_inputs)

    for m in model.modules():
        if isinstance(m, timm.models.maxxvit.MbConvBlock):
            x_train = torch.load(train_data[m].name)
            x_test = torch.load(m.filter_output.name)
            indexes = eval_function(m, conv_ratio, x_train, labels, x_test, test_labels)
            zero_weights_filter(m, indexes)

        if isinstance(m, timm.models.maxxvit.Attention2d):
            x_train = torch.load(train_data[m].name)
            x_test = torch.load(m.attention_output.name)
            indexes = eval_function(m, trans_ratio, x_train, labels, x_test, test_labels)
            zero_weights_attention(m, indexes)

    return evaluate_model()


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0, min_impurity_decrease=0.0001)


print('='*32)
print('pruning only transformers')
print('='*32)
with open('result_dt_prune_only_transformer.csv', 'w') as f:
    for iteration in range(10):
        print('iteration: ', iteration)
        model = load_model()
        top1, top2 = prune(.0, float(iteration)/10, decision_tree_importance)
        f.write(f'{iteration},{top1},{top2}\n')


print('='*32)
print('pruning only convs')
print('='*32)
with open('result_dt_prune_only_conv.csv', 'w') as f:
    for iteration in range(10):
        print('iteration: ', iteration)
        model = load_model()
        top1, top2 = prune(float(iteration)/10, .0, decision_tree_importance)
        f.write(f'{iteration},{top1},{top2}\n')


print('='*32)
print('pruning both')
print('='*32)
with open('result_dt_prune_both.csv', 'w') as f:
    for iteration in range(10):
        print('iteration: ', iteration)
        model = load_model()
        top1, top2 = prune(float(iteration)/10, float(iteration)/10, decision_tree_importance)
        f.write(f'{iteration},{top1},{top2}\n')


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, min_impurity_decrease=0.0001)

print('='*32)
print('pruning only transformers')
print('='*32)
with open('result_rf_prune_only_transformer.csv', 'w') as f:
    for iteration in range(10):
        print('iteration: ', iteration)
        model = load_model()
        top1, top2 = prune(.0, float(iteration)/10, decision_tree_importance)
        f.write(f'{iteration},{top1},{top2}\n')


print('='*32)
print('pruning only convs')
print('='*32)
with open('result_rf_prune_only_conv.csv', 'w') as f:
    for iteration in range(10):
        print('iteration: ', iteration)
        model = load_model()
        top1, top2 = prune(float(iteration)/10, .0, decision_tree_importance)
        f.write(f'{iteration},{top1},{top2}\n')


print('='*32)
print('pruning both')
print('='*32)
with open('result_rf_prune_both.csv', 'w') as f:
    for iteration in range(10):
        print('iteration: ', iteration)
        model = load_model()
        top1, top2 = prune(float(iteration)/10, float(iteration)/10, decision_tree_importance)
        f.write(f'{iteration},{top1},{top2}\n')

print('prunning_done')

# model = load_model()
# prune(.0, .5, decision_tree_importance)

# Pruning interactively
# for prune_index in range(5):
#     print('prune index: ', prune_index)
#     prune(.5/5, .5/5, decision_tree_importance)




# def prune_range():


# %%
# for m in model.modules():
#     if isinstance(m, timm.models.maxxvit.Attention2d):
#         m.old_forward = m.forward.__get__(m, type(m))
#         m.forward = zero_before_forward.__get__(m, type(m))


# output = model(example_inputs)
