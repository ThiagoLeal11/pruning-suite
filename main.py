import time

import timm
import torch

from pruning_suite.pruning import Pruning
import pruning_suite.models.coatnet as coatnet
import pruning_suite.importance.classifiers as classifiers
import pruning_suite.importance.norm as norm
import pruning_suite.evaluate as evaluate
from pruning_suite.common import PruningDataset


EXTRACT_FEATURES = {
    'timm.models.maxxvit.Attention2d': coatnet.extract_features_attention,
    'timm.models.maxxvit.MbConvBlock': coatnet.extract_features_mb_conv_block,
}

EXTRACT_WEIGHTS = {
    'timm.models.maxxvit.Attention2d': coatnet.extract_attention_weights,
    'timm.models.maxxvit.MbConvBlock': coatnet.extract_conv_weights,
}


def get_leaves(d: dict) -> float:
    if isinstance(d, dict):
        return sum([get_leaves(v) for v in d.values()])

    if isinstance(d, (int, float)):
        return float(d)

    return 0


def do_pruning(pruning_ratio, importance, is_global, interactive_steps, train_loader, test_loader, val_loader):
    # Load model
    model = timm.create_model(
        'coatnet_0_rw_224', checkpoint_path='model/coatnet_0/model_best.pth.tar', num_classes=3
    )

    # Evaluate model without pruning
    if get_leaves(pruning_ratio) == 0.0:
        top1, top2 = evaluate.top_accuracy(model, val_loader, top_k=(1, 2), device='cpu')
        print(f'Accuracy without pruning: top1={top1}, top2={top2}')
        return top1, top2

    worker = Pruning(
        model=model,
        inputs=PruningDataset(train_loader, test_loader),
        pruning_ratio=pruning_ratio,
        extract_fn=EXTRACT_FEATURES,
        ranking_fn=importance,
        prune_fn=coatnet.zero_weights,
        interactive_steps=interactive_steps,
        global_pruning=is_global,
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
    torch.set_num_threads(8)
    # Load datasets
    train_loader = load_loader(path='data/train_loader.pth')
    test_loader = load_loader(path='data/test_loader.pth')
    val_loader = load_loader(path='data/val_loader.pth')

    # Pruning definition
    pruning_ratio = {
        'timm.models.maxxvit.Attention2d': 1 / 10,
        'timm.models.maxxvit.MbConvBlock': {
            'conv2_kxk': 1 / 10,
        },
    }

    # Configs
    importance_model = classifiers.ClassifierBasedImportance(
        estimator='decision_tree', n_jobs=8
    )
    # importance_model = norm.NormBasedImportance(
    #     extract_weights_fn=EXTRACT_WEIGHTS,
    #     norm='l1',
    #     scale=True,
    # )
    is_global = False
    interactive_steps = 1

    time_start = time.time()
    do_pruning(pruning_ratio, importance_model, is_global, interactive_steps, train_loader, test_loader, val_loader)
    time_end = time.time()
    print(f'Time taken: {time_end - time_start} seconds')

    # with open('results_dt_transformer.csv', 'w') as f:
    #     for iteration in range(10):
    #         pruning_ratio = float(iteration) / 10
    #         top1, top2 = do_pruning(prune_ratio, importance_model, is_global, interactive_steps, train_loader, test_loader, val_loader)
    #         f.write(f'{pruning_ratio},{top1},{top2}\n')
    #         f.flush()


if __name__ == '__main__':
    main()
