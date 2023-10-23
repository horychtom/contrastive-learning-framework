""" This module contains the pooling layer for the transformer model. """
import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class PoolingLayer(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    :param pooling_mode_lasttoken: Perform last token pooling, see https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = "cls",
    ):
        """Initialize the pooling layer"""
        super(PoolingLayer, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_lasttoken",
        ]

        assert pooling_mode in [
            "mean",
            "max",
            "cls",
            "lasttoken",
        ]
        self.pooling_mode = pooling_mode
        self.pooling_mode_cls_token = pooling_mode == "cls"
        self.pooling_mode_max_tokens = pooling_mode == "max"
        self.pooling_mode_mean_tokens = pooling_mode == "mean"
        self.pooling_mode_lasttoken = pooling_mode == "lasttoken"
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode == "meansqrt"

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_output_dimension = word_embedding_dimension

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Pools the token embeddings by either mean or max pooling.

        Args:
                features (Dict[str, Tensor]): A dictionary containing the input features.
                        The dictionary should contain the following keys:
                        - "token_embeddings": A tensor of shape (batch_size, sequence_length, embedding_size)
                            containing the token embeddings for each token in the input sequence.
                        - "attention_mask": A tensor of shape (batch_size, sequence_length) containing the
                            attention mask for the input sequence.

        Returns:
                Dict[str, Tensor]: A dictionary containing the output features. The dictionary
                should contain the following key:
                - "sentence_embedding": A tensor of shape (batch_size, embedding_size) containing
                    the sentence embedding for each input sequence.
        """
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get(
                "cls_token_embeddings",
                token_embeddings[:, 0],
            )  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            token_embeddings[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if (
            self.pooling_mode_mean_tokens
            or self.pooling_mode_mean_sqrt_len_tokens
        ):
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded,
                1,
            )

            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
            gather_indices = (
                torch.argmin(attention_mask, 1, keepdim=False) - 1
            )  # Shape [bs]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            embedding = torch.gather(
                token_embeddings * input_mask_expanded,
                1,
                gather_indices,
            ).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sentence_embedding": output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        """Returns the dimension of sentence embeddings."""
        return self.pooling_output_dimension

    def get_config_dict(self):
        """Returns the config of the pooling layer as a dictionary."""
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        """Saves the pooling layer to the given output_path."""
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        """Loads the pooling layer from the given input_path."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return PoolingLayer(**config)
