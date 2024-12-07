import json

class Global:
    config = {}

    @staticmethod
    def init(config_path: str):
        with open(config_path, 'r') as config_file:
            Global.config = json.load(config_file)