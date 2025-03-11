import omegaconf

def load_config(config_path: str) -> omegaconf.DictConfig:
    return omegaconf.OmegaConf.load(config_path)

def save_config(config: omegaconf.DictConfig, config_path: str):
    omegaconf.OmegaConf.save(config, config_path)