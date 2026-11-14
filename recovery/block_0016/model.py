import math
import torch
from torch import nn
from torch.nn import functional as F


class Layer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                   dropout_p=self.dropout if self.training else 0.0)
        attention = attention.transpose(1, 2).reshape(b, s, d)
        x = self.norm1(x + F.dropout(self.projection(attention), self.dropout, self.training))
        ff = self.ff2(F.dropout(F.relu(self.ff1(x)), self.dropout, self.training))
        return self.norm2(x + F.dropout(ff, self.dropout, self.training))


class TransformerLM(nn.Module):
    """Causal post-LN LM, tied word embeddings, sinusoidal positions, full softmax."""
    def __init__(self, vocab_size, layers=6, d_model=256, d_ff=1024, heads=4, seq_len=512, dropout=0.3):
        super().__init__()
        if d_model % heads or d_model % 2:
            raise ValueError("d_model must be even and divisible by heads")
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Layer(d_model, d_ff, heads, dropout) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
        pos = torch.arange(seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("positions", pe, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        x = self.dropout(self.embedding(tokens) * self.scale + self.positions[:tokens.size(1)])
        for layer in self.layers:
            x = layer(x)
        return F.linear(x, self.embedding.weight)
