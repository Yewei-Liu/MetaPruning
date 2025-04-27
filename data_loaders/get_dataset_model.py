import torch
import torch.nn.functional as F
from datasets import Dataset
from data_loaders.dataset import DatasetModel
from torch.utils.data import DataLoader
from utils.logging import get_logger
from omegaconf import OmegaConf


def get_dataset_model_loader(cfg):
    model_name = cfg.dataset_model_name.split('_')[0]
    dataset_model = Dataset.load_from_disk(cfg.dataset_model_path)
    datadict = dataset_model.train_test_split(test_size=(1.0 - cfg.train_split), shuffle=False)
    model_train_dataset = DatasetModel(model_name, datadict["train"])
    model_val_dataset = DatasetModel(model_name, datadict['test'])
    model_train_loader = DataLoader(model_train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    model_val_loader = DataLoader(model_val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    logger = get_logger("get_dataset_model_loader")
    logger.info(f"train dataset model size: {len(model_train_loader)}")
    logger.info(f"val dataset model size: {len(model_val_loader)}")
    logger.info("="*50)
    return model_train_loader, model_val_loader