import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="local.env")

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

RANDOM_SEED = 321
PROJECT_NAME = "zbmath_recsys"
