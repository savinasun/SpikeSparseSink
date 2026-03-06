import yaml
import argparse


class AttrDict:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, AttrDict(value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                data[key] = value.to_dict()
            else:
                data[key] = value
        return data
    

def get_cfg(yaml_files=None):
    if yaml_files is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("yaml_files", nargs="+")
        yaml_files = parser.parse_args().yaml_files

    data = {}
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data.update(yaml.safe_load(f))
    return AttrDict(data)
