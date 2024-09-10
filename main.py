import yaml
from src import core


with open('config.yaml', 'r') as file:
    cfg_data = yaml.load(file, Loader=yaml.FullLoader)

R, t = core.get_pose(cfg_data)

print(R)
print(t)
