from torch import nn
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
from src.models.pooling import PoolingLayer
import pandas as pd
from src.data.custom_dataset import CustomTorchDataset


class EmbeddingModel(nn.Module):
    """
    A PyTorch module that represents a transformer model for computing content similarity between two pieces of text.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model to load.
        model_args (dict): A dictionary of arguments to pass to the model during initialization.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_args: dict,
    ):
        super(EmbeddingModel, self).__init__()

        self.model_args = model_args
        lm_config = AutoConfig.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
        )
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.language_model = self._load_model(
            model_name_or_path,
            lm_config,
            model_args=self.model_args,
        )

        self.pooler_layer = PoolingLayer(
            pooling_mode="mean",
            word_embedding_dimension=lm_config.hidden_size,
        )
        self.config = self.language_model.config

    def _load_model(self, model_name_or_path, config, model_args):
        """
        Loads a pre-trained transformer model from the given `model_name_or_path` and `config` using the specified `model_args`.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model to load.
            config (PretrainedConfig): The configuration object for the model.
            model_args (dict): A dictionary of arguments to pass to the model during initialization.

        Returns:
            model (PreTrainedModel): The loaded pre-trained model.
        """
        if model_args["load_quantized"]:
            quantized_model = AutoModel.from_pretrained(
                model_name_or_path,
                config=config,
                quantization_config=self.quantization_config,
                device_map="auto",
            )
            self.tokenizer.pad_token_id = 0
            return quantized_model

        return AutoModel.from_pretrained(
            model_name_or_path,
            config=config,
            device_map="auto",
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor of token ids.
            attention_mask (torch.Tensor): Input tensor of attention masks.

        Returns:
            dict: A dictionary containing token embeddings and attention masks.
        """
        output = self.language_model(
            input_ids,
            attention_mask,
            return_dict=False,
        )
        features = {}
        features.update(
            {
                "token_embeddings": output[0],
                "attention_mask": attention_mask,
            },
        )
        features = self.pooler_layer(features)
        return features

    def get_word_embedding_dimension(self) -> int:
        """Returns the dimension of the word embeddings"""
        return self.language_model.config.hidden_size

    def tokenize(self, data: pd.DataFrame):
        """Tokenizes the input data
        :param data: pd.DataFrame, input data to be tokenized
        :return:  CustomTorchDataset, tokenized dataset
        """
        tokenized = []

        def tokenize_single_(d):
            tokenized_anchor = self.tokenizer(
                d["anchor"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.model_args["max_length"],
                truncation=True,
            )
            tokenized_rec = self.tokenizer(
                d["rec"],
                return_tensors="pt",
                padding="max_length",
                max_length=self.model_args["max_length"],
                truncation=True,
            )
            return {
                "input_ids": torch.cat(
                    [
                        tokenized_rec["input_ids"],
                        tokenized_anchor["input_ids"],
                    ],
                ),
                "attention_mask": torch.cat(
                    [
                        tokenized_rec["attention_mask"],
                        tokenized_anchor["attention_mask"],
                    ],
                ),
            }

        for idx, row in data.iterrows():
            tokenized.append(tokenize_single_(row))
        labels = data.label.tolist()
        return CustomTorchDataset(tokenized, labels)
