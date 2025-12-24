import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace

config = None


@hydra.main(version_base="1.3", config_path='Conf', config_name='config')
def load_config(cfg: DictConfig):
    global config
    config = OmegaConf.to_container(cfg, resolve=True)

    # TODO: Remove [DictConfig -> Dictionary -> SimpleNamespace] conversion
    def _dict_to_namespace(d):
        return SimpleNamespace(**{k: _dict_to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})

    config = _dict_to_namespace(config)
    print(config)


if __name__ == "__main__":
    load_config()
    print(config)
    breakpoint()
