import time
import logging
import warnings
from pathlib import Path

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from torch.nn import Sequential, Conv2d, BatchNorm2d
from torch.cuda.amp import GradScaler
from tqdm import trange
import hydra
from omegaconf import OmegaConf

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch_geometric
import matplotlib.pyplot as plt
import networkx as nx
import torchvision.transforms as transforms
from torch.func import functional_call
import torchvision
import torch_pruning as tp
from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset
from utils.logging import get_logger
from utils.visualize import visualize_subgraph
from utils.convert import resnet56_graph_to_state_dict
from generate_dataset.resnet_family import resnet56
from pruning.pruner import get_pruner

from data_loaders.dataset import ResnetDataset
from nn.GNN import NNGNN
from nn.GNN2 import NNGNN2
dataset_path = './dataset/CIFAR10'

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
    root=dataset_path, train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=1000, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(
    root=dataset_path, train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=False, num_workers=4)


device = 'cuda'
# save_path = 'save/cifar10_resnet56.pth'
# speed_up = 2.0
# model = resnet56(10)
# dataset_model = Dataset.load_from_disk('dataset_model/resnet_on_CIFAR10')
# datadict = dataset_model.train_test_split(test_size=0.1, shuffle=False)
# model_train_dataset = ResnetDataset(datadict["train"])
# datadict = datadict["test"].train_test_split(test_size=0.5, shuffle=False)
# model_val_dataset = ResnetDataset(datadict['train'])
# model_test_dataset = ResnetDataset(datadict['test'])
# model_train_loader = DataLoader(model_train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
# model_val_loader = DataLoader(model_val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
# model_test_loader = DataLoader(model_test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
# origin_state_dict, info, node_index, node_features, edge_index, edge_features = next(iter(model_train_loader))
# model.load_state_dict(origin_state_dict)
def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()



def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    lr_decay_milestones,
    lr_decay_gamma=0.1,
    weight_decay=5e-4,
    device=None,
    log=False,
    verbose=False
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    if log:
        logger = get_logger("train")
    model.to(device)
    for epoch in range(epochs):
        model.train()
    
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and log and verbose:
                logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )
        
        model.eval()
        if log:
            acc, val_loss = eval(model, test_loader, device=device)
            logger.info(
                "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                    epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
                )
            )
        scheduler.step()
    train_acc, train_loss = eval(model, train_loader, device=device)
    val_acc, val_loss = eval(model, test_loader, device=device)
    if log:
        logger.info("Train Acc=%.4f  Val Acc=%.4f" % (train_acc, val_acc))
    return train_acc, train_loss, val_acc, val_loss


def meta_eval(
    model_loader,
    train_data_loader,
    metanetwork,
    device = None
):
    train_acc = []
    train_loss = []
    gt_train_acc = []
    gt_train_loss = []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet56(10).eval().to(device)
    metanetwork.eval()
    with torch.no_grad():
        for i, (origin_state_dict, info, node_index, node_features, edge_index, edge_features) in enumerate(model_loader):  
            gt_train_acc.append(info['train_acc'])
            gt_train_loss.append(info['train_loss'])
            correct = 0
            total = 0
            loss = 0
            node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
            state_dict = resnet56_graph_to_state_dict(origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
            for i, (data, target) in enumerate(train_data_loader):
                data, target = data.to(device), target.to(device)
                out = functional_call(model, state_dict, data)
                loss += F.cross_entropy(out, target, reduction="sum")
                pred = out.max(1)[1]
                correct += (pred == target).sum()
                total += len(target)
            train_acc.append((correct / total).item())
            train_loss.append((loss / total).item())
            correct = 0
            total = 0
            loss = 0
    res = {'train_acc': train_acc, 'train_loss': train_loss, 'gt_train_acc': gt_train_acc, 'gt_train_loss': gt_train_loss}
    return res




'''
To do:
1. tensorboard
2. grad scaler
'''
def meta_train(
    metanetwork,
    model_train_loader,
    data_train_loader,
    data_val_loader,
    dataset_name, 
    epochs = 100,
    lr = 0.001,
    weight_decay = 0,
    lr_decay_milestones = "40, 70",
    lr_decay_gamma = 0.1,
    pruner_reg = 1,
    eval_every_epoch = 10,
    speed_up = 2.0,
    device = None,
    log = False,
    verbose_meta_train = False,
    verbose_pruning = True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(
        metanetwork.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=lr_decay_gamma
    )
    if log:
        logger = get_logger("train")
    metanetwork.to(device)
    model = resnet56(10).eval().to(device)
    pruner = get_pruner(model, torch.ones((1, 3, 32, 32)).to(device), pruner_reg, dataset_name)

    metanetwork.eval()
    with torch.no_grad():
        res = meta_eval(model_train_loader, data_train_loader, metanetwork, device)
    if log:
        logger.info(
                    f"Train model :\n\
                    Train acc={np.mean(res['train_acc'])}({np.std(res['train_acc'])})\n\
                    Gt Train acc={np.mean(res['gt_train_acc'])}({np.std(res['gt_train_acc'])})\n\
                    Train Loss={np.mean(res['train_loss'])}({np.std(res['train_loss'])})\n\
                    Gt Train loss={np.mean(res['gt_train_loss'])}({np.std(res['gt_train_loss'])})\n\
                    Val Acc={np.mean(res['val_acc'])}({np.std(res['val_acc'])})\n\
                    Gt Val Acc={np.mean(res['gt_val_acc'])}({np.std(res['gt_val_acc'])})\n\
                    Val Loss={np.mean(res['val_loss'])}({np.std(res['val_loss'])})\n\
                    Gt Val Loss={np.mean(res['gt_val_loss'])}({np.std(res['gt_val_loss'])})\n\
                    lr={optimizer.param_groups[0]['lr']}"
                )
        
    #pruning
    if log:
        for i in range(1):
            with torch.no_grad():
                example_inputs = torch.ones((1, 3, 32, 32)).to(device)
                origin_state_dict, info, node_index, node_features, edge_index, edge_features = next(iter(model_train_loader))
                tmp_model = resnet56(10).eval().to(device)
                state_dict = origin_state_dict # resnet56_graph_to_state_dict(origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
                tmp_model = resnet56(10).eval().to(device)
                tmp_model.load_state_dict(state_dict)
            pruning(tmp_model, pruner_reg, speed_up, example_inputs, logger, data_val_loader, device, name='train_model', verbose=verbose_pruning)
            del tmp_model
        # torch.cuda.empty_cache()

    for epoch in range(epochs):
        metanetwork.train()
        for i, (origin_state_dict, info, node_index, node_features, edge_index, edge_features) in enumerate(model_train_loader):
            def onetrainstep():
                node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
                state_dict = resnet56_graph_to_state_dict(origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
                losses = 0.0
                optimizer.zero_grad()
                for j, (data, target) in enumerate(data_train_loader):
                    data, target = data.to(device), target.to(device)   
                    out = functional_call(model, state_dict, data)
                    loss = F.cross_entropy(out, target)
                    losses += loss.item()
                    loss.backward(retain_graph=True)
                model.load_state_dict(state_dict)
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad = torch.zeros_like(param)
                pruner.regularize(model)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        state_dict[name].backward(gradient = param.grad.detach().clone(), retain_graph=True)

                optimizer.step()
                
                if log and verbose_meta_train:
                    logger.info(
                        "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={}".format(
                            epoch + 1,
                            epochs,
                            i + 1,
                            len(model_train_loader),
                            losses / len(data_train_loader),
                            optimizer.param_groups[0]["lr"],
                        )
                    )
            onetrainstep()
        scheduler.step()
        if (epoch + 1) % eval_every_epoch == 0 or epoch == epochs - 1:
            metanetwork.eval()
            with torch.no_grad():
                res = meta_eval(model_train_loader, data_train_loader, data_val_loader, metanetwork, device)
            if log:
                logger.info(
                    f"Train model :\n\
                    Train acc={np.mean(res['train_acc'])}({np.std(res['train_acc'])})\n\
                    Gt Train acc={np.mean(res['gt_train_acc'])}({np.std(res['gt_train_acc'])})\n\
                    Train Loss={np.mean(res['train_loss'])}({np.std(res['train_loss'])})\n\
                    Gt Train loss={np.mean(res['gt_train_loss'])}({np.std(res['gt_train_loss'])})\n\
                    Val Acc={np.mean(res['val_acc'])}({np.std(res['val_acc'])})\n\
                    Gt Val Acc={np.mean(res['gt_val_acc'])}({np.std(res['gt_val_acc'])})\n\
                    Val Loss={np.mean(res['val_loss'])}({np.std(res['val_loss'])})\n\
                    Gt Val Loss={np.mean(res['gt_val_loss'])}({np.std(res['gt_val_loss'])})\n\
                    lr={optimizer.param_groups[0]['lr']}"
                )
            
            #pruning
            for i in range(1):
                with torch.no_grad():
                    example_inputs = torch.ones((1, 3, 32, 32)).to(device)
                    origin_state_dict, info, node_index, node_features, edge_index, edge_features = next(iter(model_train_loader))
                    tmp_model = resnet56(10).eval().to(device)
                    node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
                    state_dict = resnet56_graph_to_state_dict(origin_state_dict, node_index, node_pred, edge_index, edge_pred, device)
                    tmp_model = resnet56(10).eval().to(device)
                    tmp_model.load_state_dict(state_dict)
                pruning(tmp_model, pruner_reg, speed_up, example_inputs, logger, data_val_loader, device, name='train_model', verbose=verbose_pruning)
                del tmp_model
            # torch.cuda.empty_cache()

dataset_model_path = "dataset_model/resnet_on_CIFAR10"
dataset_model = Dataset.load_from_disk(dataset_model_path)
datadict = dataset_model.train_test_split(test_size=0.99, shuffle=False)
model_train_dataset = ResnetDataset(datadict["train"])
datadict = datadict["test"].train_test_split(test_size=0.98, shuffle=False)
model_val_dataset = ResnetDataset(datadict['train'])
model_test_dataset = ResnetDataset(datadict['test'])
model_train_loader = DataLoader(model_train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
model_val_loader = DataLoader(model_val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
model_test_loader = DataLoader(model_test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
origin_state_dict, info, node_index, node_features, edge_index, edge_features = next(iter(model_train_loader))

speed_up = 3.0
save_path = "save"
example_inputs = torch.ones((1, 3, 32, 32)).to(device)

# model = resnet56(10)
# model.load_state_dict(origin_state_dict)
model = torch.load(f"{save_path}/1.30.pth")
model.to(device)
# train_model(model, train_loader, test_loader, 100, 0.1, "40,70", device=device, log=True)
pruner = get_pruner(model, example_inputs, 0.1, 10)
base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
current_speed_up = 1
cur_model = deepcopy(model)

while current_speed_up < speed_up:
    pruner.step()

    pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    acc, loss = eval(model, test_loader, device)
    if acc < 0.9:
        torch.save(cur_model, f"{save_path}/{current_speed_up * 1.3:.2f}.pth")
        break
    current_speed_up = float(base_ops) / pruned_ops
    cur_model = deepcopy(model)
    print(current_speed_up, acc, loss)
    if pruner.current_step == pruner.iterative_steps:
        break
del pruner


