import time
import hydra
from omegaconf import OmegaConf
from datasets import Dataset
from utils.mylogging import get_logger
import os

@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg):
    logger = get_logger(name="generator")
    logger.info(f"Start generate dataset.")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    
    os.makedirs(cfg.save_path, exist_ok=True)
    data_generator = hydra.utils.instantiate(cfg.generator)
    start_time = time.time()
    dataset = Dataset.from_generator(data_generator, cache_dir=cfg.cache_path)
    dataset.save_to_disk(cfg.save_path, num_proc=cfg.num_proc)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Elapsed time: {elapsed_time / 60.0:.2f} minutes")

if __name__ == "__main__":
    main()