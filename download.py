import gdown
import os

# download goal yaml
dataset_url = 'https://drive.google.com/uc?id=' + "1OPnIktgxAhfqkBWskPRorzmday7vkvf-"
dataset_name = "goal.yaml"
path = os.path.join("./catkin_ws/src/doorgym/" + dataset_name)
gdown.download(dataset_url, output = path, quiet=False)