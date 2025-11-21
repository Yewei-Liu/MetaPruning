import os
import numpy as np
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.utils.data
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.ops as ops


# -----------------------------
# 1. Pascal VOC Dataset Wrapper
# -----------------------------

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
)


class VOCDataset(torch.utils.data.Dataset):
    """
    Wrap torchvision.datasets.VOCDetection to return samples in the format
    expected by torchvision detection models (Faster R-CNN, etc.).
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
            img = self.transforms(img)

        return img, target_out


# -----------------------------
# 2. Data Transforms & Collate
# -----------------------------

def get_transform(train: bool = True):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# -----------------------------
# 3. Build Faster R-CNN model
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


# -----------------------------
# 4. mAP evaluation (VOC07)
# -----------------------------

def voc_ap(rec, prec, use_07_metric=True):
    """Compute AP using the VOC07 11-point metric (default) or the continuous metric."""
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap

    # Continuous metric (VOC10+)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_iou(box, boxes):
    """Compute IoU between a single box and multiple boxes."""
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
):
    """
    Evaluate VOC-style mAP@IoU=0.5 (VOC07 11-point by default)
    on the given data_loader.
    """
    model.eval()

    # Collect detections and annotations per class
    all_detections = {cls: [] for cls in VOC_CLASSES}
    all_annotations = {cls: {} for cls in VOC_CLASSES}
    npos_per_class = {cls: 0 for cls in VOC_CLASSES}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, t in zip(outputs, targets):
            image_id = int(t["image_id"].item())

            # GT
            gt_boxes = t["boxes"].cpu().numpy()
            gt_labels = t["labels"].cpu().numpy()

            for box, label in zip(gt_boxes, gt_labels):
                cls_name = VOC_CLASSES[label - 1]
                if image_id not in all_annotations[cls_name]:
                    all_annotations[cls_name][image_id] = []
                all_annotations[cls_name][image_id].append(box)
                npos_per_class[cls_name] += 1

            # Detections
            det_boxes = out["boxes"].cpu().numpy()
            det_scores = out["scores"].cpu().numpy()
            det_labels = out["labels"].cpu().numpy()

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

        # Convert annotations to arrays and detection flags
        for img_id in annotations:
            annotations[img_id] = np.array(annotations[img_id])
        img_detected = {img_id: np.zeros(len(boxes)) for img_id, boxes in annotations.items()}

        # Sort detections by score
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
# 5. Training
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


def main():
    # Set this to the directory that CONTAINS VOCdevkit
    # e.g. "VOCtrainval_06-Nov-2007"
    trainval_root = "../dataset/VOCtrainval_06-Nov-2007"
    test_root = "../dataset/VOCtest_06-Nov-2007"

    ckpt_dir = "checkpoints/pretrain"
    os.makedirs(ckpt_dir, exist_ok=True)

    num_classes = 21  # 20 classes + background
    num_epochs = 10
    lr = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    batch_size = 4
    num_workers = 4

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # ------------- use train / val splits -------------
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

    # Model
    model = get_faster_rcnn_resnet50(num_classes=num_classes)
    model.to(device)

    # Optimizer & LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    best_map = 0.0
    best_model_path = os.path.join(ckpt_dir, "fasterrcnn_resnet50_voc07_best.pth")

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            print_freq=50,
        )
        lr_scheduler.step()

        print("\nEvaluating on VOC07 val...")
        mAP, _ = evaluate_map_voc(model, data_loader_val, device)

        # Save best model based on val mAP
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), best_model_path)
            print(f"New best mAP = {best_map:.4f}. Saved: {best_model_path}")

        # Also save per-epoch checkpoint
        ckpt_epoch_path = os.path.join(
            ckpt_dir, f"fasterrcnn_resnet50_voc07_epoch{epoch}.pth"
        )
        torch.save(model.state_dict(), ckpt_epoch_path)
        print(f"Saved checkpoint: {ckpt_epoch_path}\n")

    print("\nEvaluating on VOC07 test...")
    mAP, _ = evaluate_map_voc(model, data_loader_test, device)


if __name__ == "__main__":
    main()
