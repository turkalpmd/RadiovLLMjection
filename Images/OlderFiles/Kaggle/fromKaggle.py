import os
import kagglehub

# Point Kaggle API to the directory containing kaggle.json
os.environ["KAGGLE_CONFIG_DIR"] = "/home/ubuntu/RadiovLLMjection/Images"

# Download latest version
path = kagglehub.dataset_download("orvile/pmram-bangladeshi-brain-cancer-mri-dataset")

print("Path to dataset files:", path)