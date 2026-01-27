import torch
import numpy as np

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200

torch.manual_seed(42)

vocab_size = 276

def get_batch(split: str):
    """
    Generates context and target tensors. Look how the first element
    of 'y' corresponds to the second element of 'x', meaning that the
    model, given a token 117, should predict token 114.

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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int16)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int16)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

print(get_batch("train"))