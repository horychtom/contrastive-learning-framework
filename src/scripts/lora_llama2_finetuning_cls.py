from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig
import evaluate
import numpy as np


# create tokenize function
def tokenize_function(examples, tokenizer):
    # extract text
    text = examples["text"]

    # tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
    )

    return tokenized_inputs


def tokenizeTrainandValid(tokenizer, model, dataset):
    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # tokenize training and validation datasets
    return dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)


# define an evaluation function to pass into trainer later
def compute_metrics(p):
    accuracy = evaluate.load("accuracy")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy.compute(
            predictions=predictions,
            references=labels,
        ),
    }


def mainTrain():
    model_checkpoint = "Llama-2-13b"

    # define label maps
    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative": 0, "Positive": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    # load dataset
    dataset = load_dataset("ankitsatpute/zbMATHrec")

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint,
        add_prefix_space=True,
    )
    tokenized_dataset = tokenizeTrainandValid(tokenizer, model, dataset)

    # create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["q_lin"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    lr = 1e-3  # size of optimization step
    batch_size = 4  # number of examples processed per optimziation step
    num_epochs = 10  # number of times model runs through training data

    # define training arguments
    training_args = TrainingArguments(
        output_dir=model_checkpoint + "-zbRecEval",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # creater trainer object
    trainer = Trainer(
        model=model,  # our peft model
        args=training_args,  # hyperparameters
        train_dataset=tokenized_dataset["train"],  # training data
        eval_dataset=tokenized_dataset["validation"],  # validation data
        tokenizer=tokenizer,  # define tokenizer
        data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
        compute_metrics=compute_metrics,  # evaluates model using compute_metrics() function from before
    )

    # train model
    trainer.train()
