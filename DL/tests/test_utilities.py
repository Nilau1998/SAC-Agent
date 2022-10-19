import os
from utilities.config_reader import get_config

def test_get_config():
    config  = get_config(os.path.join("tests", "ressources", "test_config.yaml"))
    assert config.element == 5
    assert config.nest.nested_element == 10