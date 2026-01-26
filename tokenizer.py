from collections import Counter

def get_pair_counts(ids: list):
    c = Counter()
    for i in range(0, len(ids) - 2):
        c[(ids[i], ids[i+1])] += 1 

    return c

def merge(tokens: list, pair: tuple, idx: int):
    new_ids = []
    k = 0
    while k < len(tokens):
        bigram = tokens[k:k+2]
        if k < len(tokens) - 1 and bigram == list(pair):
            new_ids.append(idx)
            k += 2
        else:
            new_ids.append(tokens[k])
            k += 1

    return new_ids

def byte_pair_encoding(tokens: list, vocab_size):
    passes = vocab_size - 256
    merges = {}
    for i in range(passes):
        c = get_pair_counts(tokens)
        idx = 256 + i
        pair = c.most_common(1)[0][0]
        merges[pair] = idx
        tokens = merge(tokens, pair, idx)

    return merges

class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def train(self, text: str, vocab_size: int=276):
        tokens = text.encode("utf-8") 
        ids = list(tokens)
        merges = byte_pair_encoding(ids, vocab_size)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, idx in merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        print("tokens length:", len(tokens))
        print("ids length:", len(ids))
        print(f"compression ratio: {len(tokens) / len(ids):.2f}")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            c = get_pair_counts(tokens)
            pair = min(c, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)

        return tokens

    def decode(self, ids: list):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text