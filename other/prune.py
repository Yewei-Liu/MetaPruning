import os
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets import VOCDetection
import torchvision.transforms.functional as F
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.ops as ops
import argparse
import torch.nn.utils.prune as prune
from utils.pruning import progressive_pruning

# -----------------------------
# 0. Pascal VOC Classes
# -----------------------------

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)


# -----------------------------
# 1. Pascal VOC Dataset Wrapper
# -----------------------------

class VOCDataset(torch.utils.data.Dataset):
    """
    Same as in your training script: wraps VOCDetection and outputs
    samples in the format expected by torchvision detection models.
    """

    def __init__(
        self,
        root: str,
        year: str = "2007",
        image_set: str = "train",
        transforms=None,
        download: bool = False,
    ):
        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.voc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img, target = self.voc[idx]
        anno = target["annotation"]

        objs = anno.get("object", [])
        if not isinstance(objs, list):
            objs = [objs]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in objs:
            name = obj["name"]
            if name not in VOC_CLASSES:
                continue

            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASSES.index(name) + 1)
            areas.append((xmax - xmin) * (ymax - ymin))
            iscrowd.append(0)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target_out = self.transforms(img, target_out)

        return img, target_out


# -----------------------------
# 2. Data Transforms & Collate
# -----------------------------

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if "boxes" in target and target["boxes"].numel() > 0:
                _, h, w = image.shape
                boxes = target["boxes"]
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                target["boxes"] = boxes
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(train: bool = True):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(Normalize(imagenet_mean, imagenet_std))
    return Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# 3. Faster R-CNN model
# -----------------------------

def get_faster_rcnn_resnet50(num_classes: int) -> FasterRCNN:
    backbone = resnet50(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    roi_pooler = ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model


def freeze_backbone_bn(backbone: nn.Module):
    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


# -----------------------------
# 4. VOC mAP evaluation
# -----------------------------

def voc_ap(rec, prec, use_07_metric=True):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap

    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(box, boxes):
    if boxes.size == 0:
        return np.array([])

    ixmin = np.maximum(box[0], boxes[:, 0])
    iymin = np.maximum(box[1], boxes[:, 1])
    ixmax = np.minimum(box[2], boxes[:, 2])
    iymax = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    inters = iw * ih
    uni = (
        (box[2] - box[0]) * (box[3] - box[1]) +
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) -
        inters
    )

    return inters / np.maximum(uni, 1e-6)


@torch.no_grad()
def evaluate_map_voc(
    model: FasterRCNN,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_thresh: float = 0.5,
    use_07_metric: bool = True,
    score_thresh: float = 0.05,
):
    model.eval()

    all_detections = {cls: [] for cls in VOC_CLASSES}
    all_annotations = {cls: {} for cls in VOC_CLASSES}
    npos_per_class = {cls: 0 for cls in VOC_CLASSES}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, t in zip(outputs, targets):
            image_id = int(t["image_id"].item())

            gt_boxes = t["boxes"].cpu().numpy()
            gt_labels = t["labels"].cpu().numpy()

            for box, label in zip(gt_boxes, gt_labels):
                cls_name = VOC_CLASSES[label - 1]
                if image_id not in all_annotations[cls_name]:
                    all_annotations[cls_name][image_id] = []
                all_annotations[cls_name][image_id].append(box)
                npos_per_class[cls_name] += 1

            det_boxes = out["boxes"].cpu().numpy()
            det_scores = out["scores"].cpu().numpy()
            det_labels = out["labels"].cpu().numpy()

            keep = det_scores >= score_thresh
            det_boxes = det_boxes[keep]
            det_scores = det_scores[keep]
            det_labels = det_labels[keep]

            for box, score, label in zip(det_boxes, det_scores, det_labels):
                cls_name = VOC_CLASSES[label - 1]
                all_detections[cls_name].append(
                    {
                        "image_id": image_id,
                        "score": float(score),
                        "bbox": box,
                    }
                )

    aps = []
    for cls_idx, cls_name in enumerate(VOC_CLASSES):
        detections = all_detections[cls_name]
        annotations = all_annotations[cls_name]
        npos = npos_per_class[cls_name]

        if npos == 0:
            aps.append(0.0)
            print(f"{cls_name:>12}: no ground truth, AP = 0.0000")
            continue

        for img_id in annotations:
            annotations[img_id] = np.array(annotations[img_id])
        img_detected = {
            img_id: np.zeros(len(boxes)) for img_id, boxes in annotations.items()
        }

        if len(detections) == 0:
            aps.append(0.0)
            print(f"{cls_name:>12}: no detections, AP = 0.0000")
            continue

        detections = sorted(detections, key=lambda x: -x["score"])
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))

        for i, det in enumerate(detections):
            img_id = det["image_id"]
            box = det["bbox"].astype(float)

            if img_id in annotations:
                gt_boxes = annotations[img_id]
                ious = compute_iou(box, gt_boxes)
                max_iou = np.max(ious) if ious.size > 0 else 0.0
                max_idx = np.argmax(ious) if ious.size > 0 else -1

                if max_iou >= iou_thresh and img_detected[img_id][max_idx] == 0:
                    tp[i] = 1
                    img_detected[img_id][max_idx] = 1
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, 1e-6)

        ap = voc_ap(rec, prec, use_07_metric=use_07_metric)
        aps.append(ap)
        print(f"{cls_name:>12}: AP = {ap:.4f}")

    mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0
    print(f"\nVOC07 mAP@0.5 = {mAP:.4f}")
    return mAP, aps


# -----------------------------
# 5. One-epoch training (reuse)
# -----------------------------

def train_one_epoch(
    model: FasterRCNN,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    print_freq: int = 20,
):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / print_freq
            print(
                f"[Epoch {epoch}] Step [{i+1}/{len(data_loader)}] "
                f"Loss: {avg_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            running_loss = 0.0


# -----------------------------
# 6. Pruning utilities
# -----------------------------

def apply_global_unstructured_pruning(model: nn.Module, amount: float = 0.2):
    """
    Apply global L1 unstructured pruning over all Conv2d and Linear weights.
    amount: fraction of weights to prune globally (0.0 - 1.0)
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    print(f"Pruning {len(parameters_to_prune)} parameter tensors globally with amount={amount}.")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Optionally, remove reparametrization so that .weight is the pruned tensor
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    # Report overall sparsity
    total_params = 0
    total_zero = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total_params += w.numel()
            total_zero += (w == 0).sum().item()
    print(f"Global sparsity: {100.0 * total_zero / total_params:.2f}% "
          f"({total_zero}/{total_params} zeros)")


# -----------------------------
# 7. Argument parsing
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load best Faster R-CNN checkpoint, prune, and finetune on VOC07"
    )
    parser.add_argument("--index", type=int, default=1,
                        help="index used in checkpoint path (same as training script)")
    parser.add_argument("--speed_up", type=float, default=2.5,
                        help="speed-up factor for pruning (e.g., 2.5 means 60% weights pruned)")
    parser.add_argument("--finetune_epochs", type=int, default=50,
                        help="number of finetuning epochs after pruning")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate for finetuning (smaller than pretraining)")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr_decay_milestones", type=str, default="25,40",
                        help="milestones (epochs) for MultiStepLR during finetune")
    return parser.parse_args()


# -----------------------------
# 8. Main: load, prune, finetune
# -----------------------------

def main():
    args = parse_args()
    index = str(args.index)

    # Paths (match your training script structure)
    trainval_root = "../dataset/VOCtrainval_06-Nov-2007"
    test_root = "../dataset/VOCtest_06-Nov-2007"

    pretrain_ckpt_dir = f"checkpoints/{index}/pretrain"
    best_model_path = os.path.join(pretrain_ckpt_dir, "fasterrcnn_resnet50_voc07_best.pth")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best checkpoint not found: {best_model_path}")

    pruned_dir = f"checkpoints/{index}/pruned_finetune"
    os.makedirs(pruned_dir, exist_ok=True)

    num_classes = 21  # 20 classes + background

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)
    print("Loading best checkpoint from:", best_model_path)

    # Datasets & loaders
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

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Model
    model = get_faster_rcnn_resnet50(num_classes=num_classes)
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Optional: freeze BN (same as pretraining script)
    freeze_backbone_bn(model.backbone)

    # # Evaluate before pruning
    # print("\nEvaluating best (unpruned) model on VOC07 val...")
    # evaluate_map_voc(model, data_loader_val, device)

    # Apply pruning
    print("\nApplying global structured pruning...")
    speed_up, model = progressive_pruning(model, "IMAGENET", None, None, args.speed_up, "group_l2_norm_max_normalizer", False, device,
                                          False, False, None, False)
    print(f"Pruning completed. Speed up: {speed_up:.2f}")

    # Evaluate immediately after pruning (before finetune)
    print("\nEvaluating pruned (not finetuned) model on VOC07 val...")
    evaluate_map_voc(model, data_loader_val, device)

    # Finetuning setup
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    milestones = [int(ms) for ms in args.lr_decay_milestones.split(",")]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    best_map = 0.0
    best_model_path_pruned = os.path.join(
        pruned_dir, f"fasterrcnn_resnet50_voc07_pruned_best.pth"
    )

    # Finetuning loop
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

        if epoch % args.eval_every == 0:
            print("\nEvaluating pruned+finetuned model on VOC07 val...")
            mAP, _ = evaluate_map_voc(model, data_loader_val, device)

            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), best_model_path_pruned)
                print(f"New best pruned mAP = {best_map:.4f}. Saved: {best_model_path_pruned}")

        ckpt_epoch_path = os.path.join(
            pruned_dir, f"fasterrcnn_resnet50_voc07_pruned_epoch{epoch}.pth"
        )
        torch.save(model.state_dict(), ckpt_epoch_path)
        print(f"Saved pruned+finetuned checkpoint: {ckpt_epoch_path}\n")

    # Final test evaluation
    print("\nEvaluating final pruned+finetuned model on VOC07 test...")
    evaluate_map_voc(model, data_loader_test, device)


if __name__ == "__main__":
    main()
