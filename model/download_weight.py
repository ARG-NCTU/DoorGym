import gdown
from zipfile import ZipFile
import os

# Download all model

dataset_url = 'https://drive.google.com/uc?id=' + "13hc7elmhEU-9fe5K_KUALVupT519Y8EZ"
dataset_name = "./model_weight.zip"
gdown.download(dataset_url, output = dataset_name, quiet=False)
zip_file = ZipFile(dataset_name)
zip_file.extractall()
zip_file.close()
