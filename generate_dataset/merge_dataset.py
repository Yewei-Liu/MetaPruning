import time
import hydra
from omegaconf import OmegaConf
from datasets import Dataset
from utils.logging import get_logger
from datasets import load_from_disk, concatenate_datasets
import os

@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    dataset_name = cfg.name
    level = cfg.level
    path_prefix = '../dataset_model'

    # for example, if you want to merge resnet56_on_CIFAR10_0, resnet56_on_CIFAR10_1, resnet56_on_CIFAR10_2
    # you should set num to 3
    num = 4
    datasets = []
    for i in range(num):
        datasets.append(load_from_disk(os.path.join(path_prefix, f"{dataset_name}_level_{level}_{i}")))
    merged_dataset = concatenate_datasets(datasets)
    merged_dataset.save_to_disk(os.path.join(path_prefix, f"{dataset_name}_level_{level}"), num_proc=1)

if __name__ == "__main__":
    main()
