from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(data_root, batch_size=128, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    import os
    loader = get_dataloader(os.path.join(os.path.dirname(__file__), "data"))
    print(f"Batches per epoch: {len(loader)}")
    imgs, labels = next(iter(loader))
    print(f"Batch shape: {imgs.shape}, label range: [{labels.min()}, {labels.max()}]")
