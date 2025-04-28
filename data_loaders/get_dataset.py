import torchvision.transforms as transforms
import torch
import torchvision
from utils.logging import get_logger
from omegaconf import OmegaConf
import datasets
import numpy as np
import random

def get_dataset_loader(cfg, log=False):    
    if log == True:
        logger = get_logger("get_dataset_loader")
    use_seed = False
    if "seed" not in cfg:
        if log:
            logger.info("No seed for dataset loader")
    else:
        seed = cfg.seed
        if log:
            logger.info(f"Seed for dataset loader: {seed}")
        use_seed = True
    if use_seed:
        def seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        generator = torch.Generator()
        generator.manual_seed(seed)

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
        testset = torchvision.datasets.CIFAR10(
            root=cfg.dataset_path, train=False, download=True, transform=transform_test)
        if use_seed:
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, 
                worker_init_fn=seed_worker, generator=generator)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                worker_init_fn=seed_worker, generator=generator)
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
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
        testset = torchvision.datasets.CIFAR100(
            root=cfg.dataset_path, train=False, download=True, transform=transform_test)
        if use_seed:
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                worker_init_fn=seed_worker, generator=generator)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                worker_init_fn=seed_worker, generator=generator)
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
            test_loader = torch.utils.data.DataLoader(
                testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    else:
        raise NotImplementedError
    if log == True:
        logger.info(f'\n\n{OmegaConf.to_yaml(cfg)}')
        logger.info("="*50)
    return train_loader, test_loader