import argparse
import re

from easydict import EasyDict as edict
from omegaconf import OmegaConf


def process_overrides(overrides):
    """Normalize CLI overrides so `foo = bar` becomes `foo=bar`."""
    combined = " ".join(overrides)
    fixed_string = re.sub(r"(\S+)\s*=\s*(\S+)", r"\1=\2", combined)
    return re.findall(r"[^\s=]+=\S+|\S+", fixed_string)


def init_config():
    """Load the base config file and merge optional CLI overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    cli_overrides = OmegaConf.from_cli(process_overrides(args.overrides))
    config = OmegaConf.merge(config, cli_overrides)
    return edict(OmegaConf.to_container(config, resolve=True))
