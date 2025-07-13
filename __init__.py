"""
Low-Rank Adaptation (LoRA)

This is an implementation of LoRA from https://arxiv.org/abs/2106.09685

LoRA freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the transformer.
This makes it possible to efficiently fine-tune large langauge models by reducing trainable parameters by a large factor.

There's a script for training a GPT2 model with LoRA on Tiny Shakespeare dataset.
"""

import torch
import torch.nn as nn

class Linear(nn.Module):
    """
    LoRA Linear Layer

    LoRA linear layer adds a low-rank decomposition to the pre-trained weight matrix W_0 ∈ R^(d × k) of the linear layer.
    
    W_0 + Delta_W = W_0 + BA
    where B ∈ R^(d × r), A ∈ R^(r × k), and the rank r << min(d, k).

    All parameters are frozen except A and B.

    Delta_W is initialized to be zero at the beginning of the training.

    When computing the LoRA contribution, the term (x · Delta_W^T) is scaled by alpha/r, where alpha is a hyperparameter.
    Once alpha is tuned, it can remain fixed even when changing r.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 r: int, alpha: int = None):
        """
        in_features: is the number of input features of the linear layer
        out_features: is the number of output features of the linear layer
        bias: is a flag indicating if there is a bias parameter
        r: is the rank of the decomposition
        alpha: is the scaling factor
        """
        super().__init__()

        # Set alpha = r if not provided. i.e. make the scaling factor alpha/r = 1
        if alpha is None:
            alpha = r

        # The pre-trained weight W_0
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        # Freeze it
        self.weight.requires_grad = False

        if bias:
            # Bias parameter b_0 (also frozen)
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.requires_grad = False
        else:
            # No bias parameter
            self.bias = None

        # scaling factor alpha/r
        self.scaling = alpha / r
        # Matrix A ∈ R^(r × k)
        self.lora_a = nn.Parameter(torch.empty((r, in_features)))
        # Matrix B ∈ R^(d × r), we keep A and B transposed
        self.lora_b = nn.Parameter(torch.empty((out_features, r)))

        with torch.no_grad():
            # Initialize A similar to a weight matrix in a normal linear layer
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            # Initialize B to 0 so that Delta_W = BA is 0 at initialization
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor):
        # Compute x · W_0^T + b_0
        result = nn.functional.linear(x, self.weight, bias=self.bias)

        # Add (alpha/r) · x · Delta_W^T = (alpha/r) · x · (BA)^T = (alpha/r) · x · A^T · B^T
        result += (x @ self.lora_a.T @ self.lora_b.T) * self.scaling

        # Return the result
        return result


class Embedding(nn.Module):
    """
    LoRA Embedding Layer

    Similar to LoRA linear layer, this adds a low-rank decomposition to the pre-trained embedding weights matrix W_0 ∈ R^(d × k).


    W_0 + Delta_W = W_0 + BA
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 r: int, alpha: int = None):
        """

        num_embeddings: is the number of embeddings
        embedding_dim: is the number embedding dimensions
        r: is the rank of the decomposition
        alpha: is the scaling factor
        """
        super().__init__()

        # Set alpha = r if not provided (i.e., make the scaling factor alpha/r = 1)
        if alpha is None:
            alpha = r

        # The pre-trained embedding weights W_0^T (frozen)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.weight.requires_grad = False

        # Scaling factor alpha/r
        self.scaling = alpha / r
        # Matrix A ∈ R^(r × k)
        self.lora_a = nn.Parameter(torch.empty((r, num_embeddings)))
        # Matrix B ∈ R^(d × r)
        self.lora_b = nn.Parameter(torch.empty((embedding_dim, r)))

        with torch.no_grad():
            # Initialize A with a normal distribution
            nn.init.normal_(self.lora_a)
            # Initialize B to 0 so that Delta_W = BA is 0 at initialization
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor):
        # Compute the embeddings: onehot(x) · W_0
        result = nn.functional.embedding(x, self.weight)

        # Add (alpha/r) · onehot(x) · Delta_W^T = (alpha/r) · onehot(x) · A^T · B^T
        result += (nn.functional.embedding(x, self.lora_a.T) @ self.lora_b.T) * self.scaling

        # Return the result
        return result