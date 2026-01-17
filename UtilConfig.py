import os

Config = {}
root_path = os.path.abspath(os.path.dirname(__file__))
Config['root_path'] = root_path
Config['project_dataset_path'] = os.path.join(root_path, "project_data")
