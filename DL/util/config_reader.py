import os
import json
from dotmap import DotMap

def get_config(config_file):
    config_path = os.path.join("configs", config_file)
    with open(config_path) as json_config_file:
        return DotMap(json.load(json_config_file))
