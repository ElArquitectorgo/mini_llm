from model import Language_model
from tokenizer import Tokenizer
import torch
import numpy as np

batch_size = 16 # number of sequences to train in parallel
block_size = 512 # context length, how many words the model will see in each batch
max_iters = 20_000
eval_interval = 2_000
eval_iters = 200
learning_rate = 6e-4
n_embedding = 512
n_head = 8
n_layer = 8
dropout = 0.1
vocab_size = 512

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
    data = np.memmap(split + '_cervantes512.bin', dtype=np.uint16, mode='r')
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
    
model = Language_model(block_size, vocab_size, n_embedding, dropout, n_layer, n_head, device)
model = model.to(device)
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
t = Tokenizer()
t.load("tokenizer_models/cervantes512.model")
open("more.txt", "w").write(t.decode(model.generate(context, max_new_tokens=10000)[0].tolist()))