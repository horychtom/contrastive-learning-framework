"""Initialize the /wandb_ directory and log into wandb_ and hugginface."""
from config import WANDB_API_KEY
from config import HF_TOKEN
import wandb
import huggingface_hub

wandb.login(key=WANDB_API_KEY)
huggingface_hub.login(HF_TOKEN)
