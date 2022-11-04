import os
from utils.config_reader import get_config

def test_get_config():
    config = get_config(os.path.join("config.yaml"))
    assert config.test.nested_test == 5