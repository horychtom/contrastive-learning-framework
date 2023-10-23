"""This script uploads the final datasets to wandb.""" ""
from src.wandb_.wandb_client import WandbClient

wandbc = WandbClient("upload")
wandbc.upload_dataset("data/final/final_train_dataset.csv")
wandbc.upload_dataset("data/final/final_dev_dataset.csv")
wandbc.upload_dataset("data/final/final_test_dataset.csv")

wandbc.finish()
