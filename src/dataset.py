from pathlib import Path
import requests
from tokenizer import Tokenizer
import numpy as np

# https://www.gutenberg.org/ebooks/author/505

p = Path(".")
file = p / "cervantes.txt" # El Quijote + Novelas ejemplares

if not file.exists():
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(file.name, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(file.name, 'r', encoding='utf-8') as f:
    data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

t = Tokenizer()
#t.train(data, 1024)
#t.save("cervantes") # training tokens: 1_441_293 validation tokens: 161_073
#t.load("cervantes.model")
train_ids = t.encode(train_data)
val_ids = t.encode(val_data)

print("Number of training tokens:", len(train_ids))
print("Number of validation tokens:", len(val_ids))

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(p / "train.bin")
val_ids.tofile(p / "val.bin")