import torch
import torch.nn.functional as F
from utils.logging import get_logger
from copy import deepcopy
from utils.pruner import get_pruner

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

def train(
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
    verbose=False,
    method=None,
    dataset_name=None,
    reg=0.001,
    return_best=False,
    opt='sgd'
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    if method is None:
        pruner = None
    else:
        example_inputs = torch.ones((1, 3, 32, 32)).to(device)
        pruner = get_pruner(model, example_inputs, reg, dataset_name, method=method)
    if opt.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay if pruner is None else 0,
        )
        milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_decay_gamma
        )
    elif opt.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=lr, 
                                        weight_decay=weight_decay if pruner is None else 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    else:
        raise NotImplementedError(f"Optimizer {opt} not implemented")
    if log:
        logger = get_logger("train")
        train_acc, train_loss = eval(model, train_loader, device=device)
        val_acc, val_loss = eval(model, test_loader, device)
        logger.info("Before training : Train Acc=%.4f  Val Acc=%.4f" % (train_acc, val_acc))
    if return_best:
        best_acc = -1
        best_state_dict = deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
    
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            for param in model.parameters():
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)    
            if pruner is not None:
                pruner.regularize(model)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if torch.isnan(param.grad).any():
                            param.grad = torch.zeros_like(param.grad)
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and log and verbose:
                logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, train loss={:.4f}, lr={:.4f}".format(
                        epoch + 1,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )
        
        model.eval()
        val_acc, val_loss = eval(model, test_loader, device=device)
        if return_best and val_acc > best_acc :
            best_acc = val_acc
            best_state_dict = deepcopy(model.state_dict())
        if log:
            logger.info(
                "Epoch {:d}/{:d}, Val Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                    epoch + 1, epochs, val_acc, val_loss, optimizer.param_groups[0]["lr"]
                )
            )
        scheduler.step()
    if pruner is not None:
        del pruner
    if return_best:
        model.load_state_dict(best_state_dict)
    train_acc, train_loss = eval(model, train_loader, device=device)
    val_acc, val_loss = eval(model, test_loader, device=device)

    if log:
        logger.info("After training : Train Acc=%.4f  Val Acc=%.4f" % (train_acc, val_acc))
    return train_acc, train_loss, val_acc, val_loss