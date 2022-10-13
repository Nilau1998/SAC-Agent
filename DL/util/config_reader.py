import os
import json
from dotmap import DotMap

class ConfigReader():
    def __init__(self, config_path):
        self.config_path = os.path.join(config_path)

    def get_config(self):
        with open(self.config_path) as json_config_file:
            return DotMap(json.load(json_config_file))
