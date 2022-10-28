from pathlib import Path
import yaml
from dotmap import DotMap

def get_config(config_file):
    raw_config = yaml.safe_load(Path("configs", config_file).read_text())
    return DotMap(raw_config)