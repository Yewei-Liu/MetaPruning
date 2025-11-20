import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOC 20 classes (background will be class 0)
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls_name: i + 1 for i, cls_name in enumerate(VOC_CLASSES)}  # +1 for background=0


# --------- Dataset wrapper to convert VOCDetection output to detection targets ---------
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year="2007", image_set="trainval", transforms=None, download=True):
        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        # target is a dict with 'annotation'
        annotation = target["annotation"]
        objects = annotation.get("object", [])
        if not isinstance(objects, list):
            objects = [objects]

        boxes = []
        labels = []

        for obj in objects:
            difficult = int(obj.get("difficult", 0))
            # you can skip difficult objects if you want:
            # if difficult == 1: continue
            name = obj["name"]
            if name not in CLASS_TO_IDX:
                continue
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])

        if len(boxes) == 0:
            # avoid empty targets (rare, but can happen)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.tensor([])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target_out = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target_out


# --------- Transforms ---------
def get_transform(train=True):
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


# --------- Model ---------
def get_model(num_classes):
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the classifier head to match VOC (21 classes: 20 + background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# --------- Collate function for DataLoader ---------
def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    data_root = "../dataset"   # where VOC will be downloaded / stored
    batch_size = 4
    num_epochs = 10
    lr = 0.005

    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "fasterrcnn_resnet50_voc_best.pth")
    final_ckpt_path = os.path.join(save_dir, "fasterrcnn_resnet50_voc_final.pth")

    # Datasets
    dataset_train = VOCDataset(
        root=data_root,
        year="2007",
        image_set="trainval",
        transforms=get_transform(train=True),
        download=True,
    )
    dataset_test = VOCDataset(
        root=data_root,
        year="2007",
        image_set="test",
        transforms=get_transform(train=False),
        download=True,
    )

    # DataLoaders
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    data_loader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    num_classes = 21  # 20 classes + background
    model = get_model(num_classes).to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_loss = float("inf")

    # --------- Training loop ---------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in data_loader_train:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        epoch_loss /= len(data_loader_train)
        lr_scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

        # --------- Save best model checkpoint ---------
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved to {best_ckpt_path}")

    # --------- Save final model weights (for later inference) ---------
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Final model weights saved to {final_ckpt_path}")

    # --------- Simple inference on a few test images ---------
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader_test):
            images = [img.to(device) for img in images]
            outputs = model(images)  # list of dicts: boxes, labels, scores

            print(f"Image {i}, detections:")
            print(outputs[0])
            if i == 4:  # just first 5 images
                break


if __name__ == "__main__":
    main()
