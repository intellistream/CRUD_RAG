from importlib import import_module

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")
