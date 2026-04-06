from .vggsound_datamodule import VGGSoundDataModule
from .wds_datamodule import WDSDataModule


def get_datamodule_class(data_name: str):
    registry = {
        "WDSDataModule": WDSDataModule,
        "VGGSoundDataModule": VGGSoundDataModule,
    }
    if data_name not in registry:
        raise KeyError(f"Unknown datamodule: {data_name}")
    return registry[data_name]


def build_datamodule(config: dict):
    data_cfg = config["datamodule"]
    data_name = data_cfg.get("data_name", "WDSDataModule")
    return get_datamodule_class(data_name)(**data_cfg["data_config"])
