from tokenizer import Tokenizer
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

batch_size = 16 # number of sequences to train in parallel
block_size = 1024 # context length, how many words the model will see in each batch
max_iters = 20_000
eval_interval = 2_000
eval_iters = 200
learning_rate = 6e-4
n_embedding = 512
n_head = 8
n_layer = 8
dropout = 0.1
vocab_size = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def get_batch(split: str):
    """
    Generates context and target tensors. Look how the first element
    of 'y' corresponds to the second element of 'x', meaning that the
    model, given a token x[i], should predict token x[i+1].

    x = [[117, 114,  32, 259, 273,  59,  10,  66],
         [ 10,  65,  78,  71,  69,  76,  79,  58],
                                            ...]],
    y = [[114,  32, 259, 273,  59,  10,  66, 117],
         [ 65,  78,  71,  69,  76,  79,  58,  10],
                                            ...]]
    
    :param split: A string to get the data from train or val.
    :type split: str
    """
    data = np.memmap(split + '.bin', dtype=np.uint16, mode='r')
    # now we generate "batch_size" random numbers between 0 and
    # len(data) - context_length to avoid index out of bounds.
    # Example: b_s=4 -> tensor([492919, 784932, 451282, 403882])
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
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
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embedding, n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(self.dropout(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embedding):
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
    def __init__(self, n_embedding, n_head):
        super().__init__()
        head_size = n_embedding // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embedding)
        self.layer_norm_1 = nn.LayerNorm(n_embedding) 
        self.layer_norm_2 = nn.LayerNorm(n_embedding) 

    def forward(self, x):
        x = x + self.sa(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))
        return x

class Language_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.blocks = nn.Sequential(*[Block(n_embedding, n_head) for _ in range(n_layer)])
        self.final_layer_norm = nn.LayerNorm(n_embedding)
        self.lm_head = nn.Linear(n_embedding, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B, T, C) batch, block, n_embedding
        position_embeddings = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
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
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = Language_model()
model = model.to(device)
t = Tokenizer()
t.load("cervantes.model")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}")

    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
open("more.txt", "w").write(t.decode(model.generate(context, max_new_tokens=10000)[0].tolist()))