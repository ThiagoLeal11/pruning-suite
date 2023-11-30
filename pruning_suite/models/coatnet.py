import timm
import torch


def extract_features_attention(m, x, *args, **kwargs):
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
        ).transpose(-1, -2).reshape(B, -1, H, W).reshape(B, 1, -1)
        outputs.append(x)
    return {
        'qkv': torch.cat(outputs, dim=1)  # (64, 192, 14, 14)
    }


def extract_features_mb_conv_block(m, x, *args, **kwargs):
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


def extract_attention_weights(m):
    if not isinstance(m, timm.models.maxxvit.Attention2d):
        raise Exception(f'Invalid module: {m} for attention weights extraction')

    dim_head = m.dim_head
    weights = {'qkv': []}
    for hi in range(m.num_heads):
        head_size = dim_head * 3
        s = hi * head_size
        e = s + head_size

        head_weights = m.qkv.weight.data[s:e, :]
        weights['qkv'].append(head_weights)

    return {
        k: torch.stack(v)
        for k, v in weights.items()
    }


def extract_conv_weights(m):
    if not isinstance(m, timm.models.maxxvit.MbConvBlock):
        raise Exception(f'Invalid module: {m} for attention weights extraction')

    weights = {
        'conv2_kxk': m.conv2_kxk.weight.data,
    }

    return weights


def zero_weights(m, x):
    if isinstance(m, timm.models.maxxvit.Attention2d):
        x = x['qkv']
        dim_head = m.dim_head
        assert m.num_heads == len(x)

        for hi, should_prune in enumerate(x):
            if not should_prune:
                continue
            for i in range(3):  # QKV indexes
                part_idx = dim_head * i
                head_idx = hi * dim_head * 3

                s = part_idx + head_idx
                e = s + dim_head
                m.qkv.weight.data[s:e, :] = .0
                m.qkv.bias.data[s:e] = .0

    if isinstance(m, timm.models.maxxvit.MbConvBlock):
        for layer, idxs in x.items():
            layer = m.__getattr__(layer)
            for fi, should_prune in enumerate(idxs):
                if not should_prune:
                    continue

                if isinstance(layer, timm.layers.squeeze_excite.SEModule):
                    layer.fc2.weight.data[fi, :] = .0
                    continue

                layer.weight.data[fi, :] = .0
