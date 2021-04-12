import torch
from torch import nn
from os.path import join
import numpy as np
import copy

from Attention import MultiHeadAttention

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_dim, heads, dropout_rate):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff = FeedForward(dim, mlp_dim, dropout_rate)
        self.mha = MultiHeadAttention(dim, heads)

        self.dim = dim

    def forward(self, x):

        h = x
        x = self.attention_norm(x)
        x = self.mha(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ff(x)
        x = x + h
        return x

    def load_from(self, weights, n_block):
            
        with torch.no_grad():
            ROOT = f"Transformer/encoderblock_{n_block}"
                    
            query_weight = torch.from_numpy(weights[join(ROOT, ATTENTION_Q, "kernel")]).view(self.dim, self.dim).t()
            key_weight = torch.from_numpy(weights[join(ROOT, ATTENTION_K, "kernel")]).view(self.dim, self.dim).t()
            value_weight = torch.from_numpy(weights[join(ROOT, ATTENTION_V, "kernel")]).view(self.dim, self.dim).t()
            out_weight = torch.from_numpy(weights[join(ROOT, ATTENTION_OUT, "kernel")]).view(self.dim, self.dim).t()
            query_bias = torch.from_numpy(weights[join(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = torch.from_numpy(weights[join(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = torch.from_numpy(weights[join(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = torch.from_numpy(weights[join(ROOT, ATTENTION_OUT, "bias")]).view(-1)
            mlp_weight_0 = torch.from_numpy(weights[join(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = torch.from_numpy(weights[join(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = torch.from_numpy(weights[join(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = torch.from_numpy(weights[join(ROOT, FC_1, "bias")]).t()
            att_norm_weight = torch.from_numpy(weights[join(ROOT, ATTENTION_NORM, "scale")])
            att_norm_bias = torch.from_numpy(weights[join(ROOT, ATTENTION_NORM, "bias")])
            ffn_norm_weight = torch.from_numpy(weights[join(ROOT, MLP_NORM, "scale")])
            ffn_norm_bias = torch.from_numpy(weights[join(ROOT, MLP_NORM, "bias")])

            self.mha.linear_q.weight.copy_(query_weight)
            self.mha.linear_q.bias.copy_(query_bias)
            self.mha.linear_k.weight.copy_(key_weight)
            self.mha.linear_k.bias.copy_(key_bias)
            self.mha.linear_v.weight.copy_(value_weight)
            self.mha.linear_v.bias.copy_(value_bias)
            self.mha.linear_o.weight.copy_(out_weight)
            self.mha.linear_o.bias.copy_(out_bias)
            self.ff.fc1.weight.copy_(mlp_weight_0)
            self.ff.fc1.bias.copy_(mlp_bias_0)
            self.ff.fc2.weight.copy_(mlp_weight_1)
            self.ff.fc2.bias.copy_(mlp_bias_1)
            self.attention_norm.weight.copy_(att_norm_weight)
            self.attention_norm.bias.copy_(att_norm_bias)
            self.ffn_norm.weight.copy_(ffn_norm_weight)
            self.ffn_norm.bias.copy_(ffn_norm_bias)

class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate,
        warmed_up
    ):
        super(TransformerModel, self).__init__()

        self.layer = nn.ModuleList()
        for n_block in range(depth):
            layer = Block(dim, mlp_dim, heads, dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        return hidden_states
