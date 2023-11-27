import torch
from timm.data import create_dataset, create_loader


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


def save_loader(loader, path: str):
    x, y = next(iter(loader))
    torch.save((x, y), path)


def main():
    train_loader = new_loader(root='data/nhl/', split='train', batch_size=128)  # 261
    test_loader = new_loader(root='data/nhl/', split='test', batch_size=64)  # 84
    val_loader = new_loader(root='data/nhl/', split='val', batch_size=29, is_training=False)

    save_loader(train_loader, 'data/train_loader.pth')
    save_loader(test_loader, 'data/test_loader.pth')
    save_loader(val_loader, 'data/val_loader.pth')


if __name__ == '__main__':
    main()
