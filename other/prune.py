import os
import argparse
from typing import Tuple, Dict

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.utils.prune as prune

from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.ops as ops
from other.voc_frcnn import evaluate_map_voc, train_one_epoch, collate_fn, get_faster_rcnn_resnet50, VOCDataset, get_transform
from utils.convert import state_dict_to_model, state_dict_to_graph, graph_to_model



def parse_args():
    parser = argparse.ArgumentParser(description="Prune and finetune Faster R-CNN ResNet50 on VOC07")

    parser.add_argument("--trainval-root", type=str, default="../dataset/VOCtrainval_06-Nov-2007",
                        help="Path to VOC trainval root (folder containing VOCdevkit)")
    parser.add_argument("--test-root", type=str, default="../dataset/VOCtest_06-Nov-2007",
                        help="Path to VOC test root")

    parser.add_argument("--ckpt-path", type=str, default="checkpoints/pretrain/fasterrcnn_resnet50_voc07_best.pth",
                        help="Path to the best pretrained checkpoint to load")

    parser.add_argument("--output-dir", type=str, default="checkpoints/pruned",
                        help="Directory to save pruned/finetuned checkpoints")

    parser.add_argument("--prune-amount", type=float, default=0.3,
                        help="Fraction of Conv/Linear weights to globally prune (0.0 ~ 1.0)")
    parser.add_argument("--finetune-epochs", type=int, default=5,
                        help="Number of finetuning epochs after pruning")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for finetuning (smaller than initial training)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Weight decay")

    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Evaluate mAP on val every N epochs")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_classes = 21  # 20 classes + background
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Datasets & DataLoaders
    dataset_train = VOCDataset(
        root=args.trainval_root,
        year="2007",
        image_set="train",
        transforms=get_transform(train=True),
        download=False,
    )

    dataset_val = VOCDataset(
        root=args.trainval_root,
        year="2007",
        image_set="val",
        transforms=get_transform(train=False),
        download=False,
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Build model and load checkpoint
    print(f"Building model and loading checkpoint from: {args.ckpt_path}")
    model = get_faster_rcnn_resnet50(num_classes=num_classes)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device)

    # Evaluate before pruning (optional but useful)
    print("\nEvaluating baseline (before pruning) on VOC07 val...")
    baseline_map, _ = evaluate_map_voc(model, data_loader_val, device)
    print(f"Baseline mAP@0.5 (val) = {baseline_map:.4f}")

    metanetwork = torch.load(metanetwork.pth, weights_only=False)
    if isinstance(metanetwork, list):
        metanetwork = metanetwork["model"]
    metanetwork.eval().to(device)
    origin_state_dict = model.state_dict()
    model_name = "resnet50_detection"
    node_index, node_features, edge_index, edge_features = state_dict_to_graph(model_name, origin_state_dict)
    node_pred, edge_pred = metanetwork.forward(node_features.to(device), edge_index.to(device), edge_features.to(device))
    model = graph_to_model(model_name, origin_state_dict, node_index, node_pred, edge_index, edge_pred)

    # Evaluate immediately after pruning (without finetuning)
    print("\nEvaluating pruned (before finetuning) on VOC07 val...")
    pruned_map, _ = evaluate_map_voc(model, data_loader_val, device)
    print(f"Pruned mAP@0.5 (val) before finetune = {pruned_map:.4f}")

    # Finetune pruned model
    print("\nStarting finetuning on pruned model...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    best_finetune_map = 0.0
    best_finetune_path = os.path.join(args.output_dir, "fasterrcnn_resnet50_voc07_pruned_best.pth")

    for epoch in range(1, args.finetune_epochs + 1):
        train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            print_freq=50,
        )
        lr_scheduler.step()

        if epoch % args.eval_interval == 0:
            print("\nEvaluating pruned+finetuned model on VOC07 val...")
            mAP, _ = evaluate_map_voc(model, data_loader_val, device)
            print(f"[Epoch {epoch}] Pruned+finetuned mAP@0.5 (val) = {mAP:.4f}")

            if mAP > best_finetune_map:
                best_finetune_map = mAP
                torch.save(model.state_dict(), best_finetune_path)
                print(f"New best pruned+finetuned mAP = {best_finetune_map:.4f}. Saved: {best_finetune_path}")

        # Save per-epoch checkpoint (optional)
        epoch_ckpt_path = os.path.join(
            args.output_dir,
            f"fasterrcnn_resnet50_voc07_pruned_epoch{epoch}.pth",
        )
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"Saved pruned+finetuned checkpoint for epoch {epoch}: {epoch_ckpt_path}\n")

    # After finetuning, remove pruning re-param and save a clean final model
    remove_pruning(parameters_to_prune)
    final_ckpt_path = os.path.join(args.output_dir, "fasterrcnn_resnet50_voc07_pruned_finetuned_final.pth")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"\nSaved final pruned & finetuned model (with pruning made permanent) to: {final_ckpt_path}")


if __name__ == "__main__":
    main()
