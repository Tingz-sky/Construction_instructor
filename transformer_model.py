# coding:utf-8
'''
**************************************************
@File   ：1724 -> transformer
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , 
@Date   ：2024/2/5 18:04
**************************************************
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = attention / (self.embed_size ** (1/2))
        attention = torch.softmax(attention, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class PositionwiseFeedforward(nn.Module):
    def __init__(self, embed_size, ff_dim):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = PositionwiseFeedforward(embed_size, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerNano_base(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, ff_dim, dropout, max_length):
        super(TransformerNano_base, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.words_embed = nn.Embedding(vocab_size, embed_size)
        self.position_embed = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, ff_dim) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).repeat(N, 1).to(self.device)

        out = self.dropout(self.words_embed(x) + self.position_embed(positions))

        for layer in self.layers:
            out = layer(out, out, out)

        out = self.fc_out(out)
        return out

class TransformerNano(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len):
        super(TransformerNano, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # Adjust positional encoding to match input size
        seq_len = x.size(1)
        if self.positional_encoding.size(1) < seq_len:
            # Extend positional encoding if needed
            self.positional_encoding = nn.Parameter(torch.cat(
                [self.positional_encoding, torch.randn(1, seq_len - self.positional_encoding.size(1), self.embed_dim)],
                dim=1))
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        x = self.embed(x) + pos_encoding
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

