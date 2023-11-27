import time

import timm
import torch

from pruning_suite.pruning import Pruning, PruningDataset
import pruning_suite.models.coatnet as coatnet
import pruning_suite.importance.vanilla as sklearn
import pruning_suite.evaluate as evaluate


EXTRACT_FEATURES = {
    'timm.models.maxxvit.Attention2d': coatnet.extract_features_attention,
    # 'timm.models.maxxvit.MbConvBlock': coatnet.extract_features_mb_conv_block,
}


def do_pruning(pruning_ratio, train_loader, test_loader, val_loader):
    # Load model
    model = timm.create_model(
        'coatnet_0_rw_224', checkpoint_path='model/coatnet_0/model_best.pth.tar', num_classes=3
    )

    # Evaluate model without pruning
    if pruning_ratio == 0:
        top1, top2 = evaluate.top_accuracy(model, val_loader, top_k=(1, 2), device='cpu')
        print(f'Accuracy without pruning: top1={top1}, top2={top2}')
        return top1, top2

    prune_ratio = {
        'timm.models.maxxvit.Attention2d': pruning_ratio,
        'timm.models.maxxvit.MbConvBlock': {
            'conv2_kxk': 0,
        },
    }

    importance = sklearn.ParallelFeatureImportance(sklearn.decision_tree_importance, workers=8)
    worker = Pruning(
        model=model,
        inputs=PruningDataset(train_loader, test_loader),
        pruning_ratio=prune_ratio,
        extract_fn=EXTRACT_FEATURES,
        ranking_fn=importance,
        prune_fn=coatnet.zero_weights,
        interactive_steps=2,
        global_pruning=False,
        device='cpu',
    )
    worker.prune()

    # Evaluate model before pruning
    top1, top2 = evaluate.top_accuracy(model, val_loader, top_k=(1, 2), device='cpu')
    print(f'Accuracy after pruning: top1={top1}, top2={top2}')
    return top1, top2


def load_loader(path: str):
    data = torch.load(path)
    return [data]


def main():
    # Load datasets
    train_loader = load_loader(path='data/train_loader.pth')
    test_loader = load_loader(path='data/test_loader.pth')
    val_loader = load_loader(path='data/val_loader.pth')

    time_start = time.time()
    do_pruning(0.5, train_loader, test_loader, val_loader)
    time_end = time.time()
    print(f'Time taken: {time_end - time_start} seconds')

    # with open('results_dt_transformer.csv', 'w') as f:
    #     for iteration in range(10):
    #         pruning_ratio = float(iteration) / 10
    #         top1, top2 = do_pruning(pruning_ratio, train_loader, test_loader, val_loader)
    #         f.write(f'{pruning_ratio},{top1},{top2}\n')
    #         f.flush()


if __name__ == '__main__':
    main()
