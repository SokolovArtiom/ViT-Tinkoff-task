import torch
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils import data
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    DistributedSampler,
    SequentialSampler,
)

def get_dataset(dataset, img_size, batch_size, num_workers):

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if dataset == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=transform_test,
            )
        )

    elif dataset == "cifar100":
        trainset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = (
            datasets.CIFAR100(
                root="./data",
                train=False,
                download=True,
                transform=transform_test,
            )
        )
    elif dataset == "places365":
        trainset = datasets.Places365(
            root="./data", split="train-standard", small=True, download=True, transform=transform_train
        )
        testset = datasets.Places365(
            root="./data", split="val", small=True, download=True, transform=transform_train
        )
    elif dataset == "flowers":
        data_dir = "data/flower_data"
        data_transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['train', 'valid']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
        return dataloaders["train"], dataloaders["valid"]

    train_sampler = (
        RandomSampler(trainset)
    )
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = (
        DataLoader(
            testset,
            sampler=test_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        if testset is not None
        else None
    )

    return train_loader, test_loader

