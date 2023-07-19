import torch
from torch import Tensor
from torch.nn import Module, TransformerEncoder, Parameter, ModuleList
from torch.nn import Linear, Embedding, Dropout, TransformerEncoderLayer

import math

import numpy as np

class PropertyEmbedding(Module):

    def __init__(self, prop_dims: list, prop_lengths: list) -> None:
        super().__init__()

        self.PropertyEmbeddings = ModuleList([Embedding(prop_lengths[i], prop_dims[i]) for i in range(len(prop_dims))])
        self.init_weights()

    
    def forward(self, prop_ids: Tensor) -> Tensor:

        if prop_ids.size(-1) != len(self.PropertyEmbeddings):
            raise RuntimeError("the number of properties does not match the number of property embeddings")
        
        prop_embeddings = [self.PropertyEmbeddings[i](prop_ids[...,i]) for i in range(prop_ids.size(-1))]
        prop_embeddings = torch.concat(prop_embeddings, dim=-1)

        return prop_embeddings
    
    def init_weights(self, r: float = 0.2) -> None:
        for property_embedding in self.PropertyEmbeddings:
            rand_weight = 2*torch.rand_like(property_embedding.weight) - 1
            property_embedding.weight = Parameter(r * rand_weight)

class PositionalEncoding(Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        n = 10_000.0

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(n) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # scale values by dividing my sqrt(dim)
        pe = pe * pow(d_model, -2.)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        if x.dim() == 2:
            x = x.unsqueeze(0)
        # x size: [# of batches, seq_len, dim]

        x = x + self.pe[:x.size(-2)]
        x = self.dropout(x)
        return x.squeeze(0)
    

class Vector2PropertyClassification(Module):

    def __init__(self, property_lengths: list, input_dim: int) -> None:
        super().__init__()

        self.property_lengths = property_lengths
        self.full_prop_dim = np.sum(property_lengths)

        self.classifier = Linear(input_dim,self.full_prop_dim)


    def forward(self, x: Tensor) -> Tensor:
        # x size: [# of batches, seq_len, dim]

        x = self.classifier(x)

        classifications = []
        start_idx = 0
        for sz in self.property_lengths:
            classifications.append(x[...,start_idx:start_idx+sz])
            start_idx += sz

        return classifications
    

class TransformerModel(Module):
    def __init__(self, prop_dims: list, prop_lengths: list, num_logits: int, dropout: float = 0.1, dim_feedforward: int = 1024, nhead: int = 8, max_len:int = 10_000, num_layers: int = 6) -> None:
        super().__init__()


        d_model = np.sum(prop_dims)


        self.property_embedding = PropertyEmbedding(prop_dims=prop_dims, prop_lengths=prop_lengths)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                            dim_feedforward=dim_feedforward, dropout=dropout, 
                                                            batch_first=True, norm_first=True, activation="gelu")
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)

        self.classifier = Linear(d_model, num_logits)


    def forward(self, src: Tensor) -> None:

        mask = self.generate_square_subsequent_mask(src.size(-2), src.device) # get src seq len

        embeddings = self.property_embedding(src)
        positional_embeddings = self.positional_encoding(embeddings)
        encoded = self.transformer_encoder(positional_embeddings, mask=mask)

        classified = self.classifier(encoded)

        return classified
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
        return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)