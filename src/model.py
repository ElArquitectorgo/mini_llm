import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embedding, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        attention = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, T)
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        attention = F.softmax(attention, dim=-1) # (B, T, T)
        attention = self.dropout(attention)

        v = self.value(x) # (B, T, C)
        out = attention @ v # (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size, n_embedding, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embedding, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(self.dropout(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embedding, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.ReLU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embedding, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embedding // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embedding, block_size, dropout)
        self.ffwd = FeedForward(n_embedding, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embedding) 
        self.layer_norm_2 = nn.LayerNorm(n_embedding) 

    def forward(self, x):
        x = x + self.sa(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))
        return x

class Language_model(nn.Module):
    def __init__(self, block_size, vocab_size, n_embedding, dropout, n_layer, n_head, device):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embedding = n_embedding
        self.dropout = dropout
        self.n_layer = n_layer
        self.n_head = n_head
        self.device = device

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embedding)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embedding)
        self.blocks = nn.Sequential(*[Block(self.n_embedding, self.n_head, self.block_size, self.dropout) for _ in range(self.n_layer)])
        self.final_layer_norm = nn.LayerNorm(self.n_embedding)
        self.lm_head = nn.Linear(self.n_embedding, self.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B, T, C) batch, block, n_embedding
        position_embeddings = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = token_embeddings + position_embeddings # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.final_layer_norm(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx