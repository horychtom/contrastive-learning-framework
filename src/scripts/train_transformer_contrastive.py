""" Trains a transformer model with contrastive loss on a given dataset. """
from src.models.transformer_model import EmbeddingModel
from src.models.contrastive_loss import ContrastiveLoss
from src.wandb_.wandb_client import WandbClient
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=4,
)

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import pandas as pd


# initiate wandb client with run name
model_name = "BAAI/llm-embedder"
run_name = model_name.split("/")[-1] + "-fine-tuning"
wandbc = WandbClient(run_name=run_name)

# load datasets from wandb
train_dataset = pd.read_csv(wandbc.load_dataset("final_train_dataset"))
dev_dataset = pd.read_csv(wandbc.load_dataset("final_dev_dataset"))

# load model and tokenizer
model_args = {
    "max_length": 512,
    "load_quantized": True,
    "pooling_mode": "mean",
}
embedding_model = EmbeddingModel(model_name, model_args=model_args)
tokenized_train = embedding_model.tokenize(train_dataset)
tokenized_dev = embedding_model.tokenize(dev_dataset)

# load loss model
loss_model = ContrastiveLoss(embedding_model)
new_model = get_peft_model(loss_model, peft_config)

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./checkpoints",
    per_device_eval_batch_size=2,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_total_limit=3,
    evaluation_strategy="steps",
    logging_steps=5,
    eval_steps=5,
    save_steps=5,
    disable_tqdm=False,
    weight_decay=0.1,
    learning_rate=4e-5,
    run_name=run_name,
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    load_best_model_at_end=True,
    remove_unused_columns=False,
)
trainer = Trainer(
    new_model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=embedding_model.tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=embedding_model.tokenizer),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.0,
        ),
    ],
)
trainer.train()

# push model and tokenizer to huggingface hub
loss_model.model.language_model.push_to_hub(
    "horychtom/" + model_name.split("/")[-1] + "-zbmath",
)
loss_model.model.tokenizer.push_to_hub(
    "horychtom/" + model_name.split("/")[-1] + "-zbmath",
)
wandbc.finish()
