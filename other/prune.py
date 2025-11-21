import os
import argparse

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.prune as prune

# ==============================
# Import all stuff from your file
# ==============================
# !!! IMPORTANT !!!
# Replace `your_training_script` with the filename that contains
# VOCDataset, get_transform, collate_fn, get_faster_rcnn_resnet50,
# freeze_backbone_bn, evaluate_map_voc, train_one_epoch, VOC_CLASSES.
#
# Example: if your original file is called `train_voc_frcnn.py`, then:
# from train_voc_frcnn import ...
from voc_frcnn import (
    VOCDataset,
    get_transform,
    collate_fn,
    get_faster_rcnn_resnet50,
    freeze_backbone_bn,
    evaluate_map_voc,
    train_one_epoch,
    VOC_CLASSES,
)


# =========================
# 1. Pruning helper function
# =========================

def apply_global_conv_pruning(model: nn.Module, amount: float = 0.3):
    """
    Globally prune Conv2d weights in the whole model using L1 unstructured pruning.

    amount:
        fraction of parameters to prune globally (e.g., 0.3 = 30%)
    """
    parameters_to_prune = []

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, "weight"))

    print(f"[Pruning] Number of Conv2d layers to prune: {len(parameters_to_prune)}")
    if len(parameters_to_prune) == 0:
        print("[Pruning] WARNING: no Conv2d modules found to prune.")
        return model

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make pruning permanent (remove the pruning re-parametrization, keep zeros)
    for m, _ in parameters_to_prune:
        try:
            prune.remove(m, "weight")
        except ValueError:
            # already removed / not pruned
            pass

    print(f"[Pruning] Global L1 unstructured pruning applied with amount={amount}.")
    return model


# ============================
# 2. Prune + finetune + test
# ============================

def prune_and_finetune(
    index: int = 0,
    prune_amount: float = 0.3,
    finetune_epochs: int = 20,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 4,
    num_workers: int = 4,
):
    """
    1) Load best pre-trained checkpoint.
    2) Apply pruning.
    3) Finetune on train split, picking best on val split.
    4) Evaluate best pruned model on test split.
    """

    index = str(index)

    # Same dataset roots as in your original training script
    trainval_root = "../dataset/VOCtrainval_06-Nov-2007"
    test_root = "../dataset/VOCtest_06-Nov-2007"

    # ---------- Checkpoint paths ----------

    pretrain_ckpt_dir = f"checkpoints/{index}/pretrain"
    pretrain_best_ckpt = os.path.join(
        pretrain_ckpt_dir, "fasterrcnn_resnet50_voc07_best.pth"
    )

    if not os.path.isfile(pretrain_best_ckpt):
        raise FileNotFoundError(
            f"Cannot find pretrain checkpoint: {pretrain_best_ckpt}\n"
            "Make sure you ran your original training script first."
        )

    pruned_ckpt_dir = f"checkpoints/{index}/pruned"
    os.makedirs(pruned_ckpt_dir, exist_ok=True)
    pruned_best_ckpt = os.path.join(
        pruned_ckpt_dir, "fasterrcnn_resnet50_voc07_pruned_best.pth"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # ---------- Datasets and loaders ----------

    dataset_train = VOCDataset(
        root=trainval_root,
        year="2007",
        image_set="train",
        transforms=get_transform(train=True),
        download=False,
    )

    dataset_val = VOCDataset(
        root=trainval_root,
        year="2007",
        image_set="val",
        transforms=get_transform(train=False),
        download=False,
    )

    dataset_test = VOCDataset(
        root=test_root,
        year="2007",
        image_set="test",
        transforms=get_transform(train=False),
        download=False,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # ---------- Build model & load pretrained weights ----------

    num_classes = len(VOC_CLASSES) + 1  # 20 + background
    model = get_faster_rcnn_resnet50(num_classes=num_classes)

    print(f"Loading pretrain checkpoint from: {pretrain_best_ckpt}")
    state_dict = torch.load(pretrain_best_ckpt, map_location="cpu")
    model.load_state_dict(state_dict)

    # Freeze BN in backbone (same as your original script)
    freeze_backbone_bn(model.backbone)

    # ---------- Apply pruning ----------

    model = apply_global_conv_pruning(model, amount=prune_amount)
    model.to(device)

    # ---------- Finetune settings ----------

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Simple step scheduler: decay LR at half and 3/4 of finetune epochs
    if finetune_epochs >= 4:
        milestones = [finetune_epochs // 2, (3 * finetune_epochs) // 4]
    else:
        milestones = [finetune_epochs]  # effectively no decay for very small epoch counts

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    best_map = 0.0

    # ---------- Finetune loop (train + val eval) ----------

    for epoch in range(1, finetune_epochs + 1):
        print(f"\n=== Finetune Epoch {epoch}/{finetune_epochs} ===")
        train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            print_freq=50,
        )
        lr_scheduler.step()

        print("\nEvaluating pruned model on VOC07 val...")
        mAP, _ = evaluate_map_voc(model, data_loader_val, device)

        # Save best on val mAP
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), pruned_best_ckpt)
            print(f"[Finetune] New best mAP = {best_map:.4f}. Saved: {pruned_best_ckpt}")

        # Optional: per-epoch checkpoint (uncomment if you want them)
        # ckpt_epoch_path = os.path.join(
        #     pruned_ckpt_dir, f"fasterrcnn_resnet50_voc07_pruned_epoch{epoch}.pth"
        # )
        # torch.save(model.state_dict(), ckpt_epoch_path)
        # print(f"[Finetune] Saved checkpoint: {ckpt_epoch_path}")

    print(f"\nBest pruned val mAP = {best_map:.4f}")
    print(f"Loading best pruned checkpoint from: {pruned_best_ckpt}")
    best_state_dict = torch.load(pruned_best_ckpt, map_location="cpu")
    model.load_state_dict(best_state_dict)
    model.to(device)

    # ---------- Final evaluation on test ----------

    print("\nEvaluating best pruned+finetuned model on VOC07 test...")
    test_mAP, class_aps = evaluate_map_voc(model, data_loader_test, device)
    print(f"\nFinal Test mAP (pruned+finetuned) = {test_mAP:.4f}")


# ====================
# 3. CLI entry point
# ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune and finetune Faster R-CNN ResNet50 on VOC07"
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--prune-amount",
        type=float,
        default=0.3,
        help="Fraction of Conv2d weights to prune globally (0.0–1.0)",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=20,
        help="Number of finetuning epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for finetuning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Finetuning batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prune_and_finetune(
        index=args.index,
        prune_amount=args.prune_amount,
        finetune_epochs=args.finetune_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
