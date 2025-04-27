import torchvision.transforms as transforms
import torch
import torchvision
from utils.logging import get_logger
from omegaconf import OmegaConf
import datasets

def get_dataset_loader(cfg, log=False):    
    if cfg.dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.2010, 0.1971)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.2010, 0.1971)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=cfg.dataset_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR10(
            root=cfg.dataset_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    elif cfg.dataset_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=cfg.dataset_path, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        testset = torchvision.datasets.CIFAR100(
            root=cfg.dataset_path, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    # elif cfg.dataset_name == 'IMAGENET':
    #     train_transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     val_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    #     train_dataset = datasets.ImageFolder(
    #         root='/path/to/imagenet/train',
    #         transform=train_transform
    #     )

    #     val_dataset = datasets.ImageFolder(
    #         root='/path/to/imagenet/val',
    #         transform=val_transform
    #     )
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=cfg.batch_size,
    #         shuffle=True,
    #         num_workers=cfg,
    #         pin_memory=True
    #     )
    #     val_loader = torch.utils.data.DataLoader(
    #         val_dataset,
    #         batch_size=BATCH_SIZE,
    #         shuffle=False,
    #         num_workers=NUM_WORKERS,
    #         pin_memory=True
    #     )
    else:
        raise NotImplementedError
    if log == True:
        logger = get_logger("get_dataset_loader")
        logger.info(f'\n\n{OmegaConf.to_yaml(cfg)}')
        logger.info("="*50)
    return train_loader, test_loader