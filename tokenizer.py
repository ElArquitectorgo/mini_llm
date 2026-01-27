from collections import Counter
import regex as re

def get_pair_counts(ids: list):
    """
    Return a counter object that contains the frequency of every consecutive pair of elements.
    Example: [1,2,9,9,1,2] -> Counter({(1, 2): 2, (2, 9): 1, (9, 9): 1, (9, 1): 1})

    :param ids: List of integers.
    :type ids: list
    """
    c = Counter()
    for i in range(0, len(ids) - 1):
        c[(ids[i], ids[i+1])] += 1 

    return c

def merge(tokens: list, pair: tuple, idx: int):
    """
    Replace all occurrences of pair in tokens with the new integer idx.
    Example: merge([1,2,9,9,1,2], (1,2), 256) -> [256, 9, 9, 256]
    
    :param tokens: List of integers.
    :type tokens: list
    :param pair: The pair that will be replaced.
    :type pair: tuple
    :param idx: The number that will replace the pair.
    :type idx: int
    """
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

def byte_pair_encoding(ids: list, vocab_size: int=256):
    """
    Apply the Byte Pair Encoding algorithm to iteratively merge 
    the most common pairs to create new tokens.
    Returns a dictionary with the merges produced.
    Example: bpe([1,2,9,9,1,2], 258) -> {(1, 2): 256, (256, 9): 257}
    
    :param ids: List of integers.
    :type ids: list
    :param vocab_size: This number determines the number of passes BPE is going to do.
    The default value if 256, meaning no passes will be executed. Since we already
    have a default dictionary of 256 tokens (utf-8), vocab_size needs to be greater
    in order to reduce the size of tokens.
    :type vocab_size: int
    """
    assert vocab_size >= 256
    passes = vocab_size - 256
    merges = {}
    for i in range(passes):
        stats = []
        for chunk_ids in ids:
            c = get_pair_counts(chunk_ids).most_common(1)
            if c:
                stats.append(c[0])
        pair = max(stats, key=lambda x: x[1])[0]
        idx = 256 + i
        merges[pair] = idx
        ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

    return merges

class Tokenizer:
    """
    This class provides core functionality for training a tokenizer and encoding/decoding text.
    """
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        # https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py -> gpt4 split pattern
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text: str, vocab_size: int=276):
        """
        Trains the Tokenizer from a text string. Generates a merges dictionary
        to encode future text strings and a vocabulary dictionary to decode
        previously encoded text.
        
        :param text: A string that will be used to train the Tokenizer.
        :type text: str
        :param vocab_size: The desired final vocabulary size.
        :type vocab_size: int
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        merges = byte_pair_encoding(ids, vocab_size)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, idx in merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def encode_chunk(self, text: str):
        """
        Codifies text into a list of integers.
        
        :param text: Text to encode.
        :type text: str
        """
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            c = get_pair_counts(tokens)
            pair = min(c, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens

    def encode(self, text: str):
        """
        Codifies text into a list of integers. The text is firts splitted into
        chunks with the regex pattern. Then we encoded those chunks separately,
        and join the results in ids.
        
        :param text: Text to encode.
        :type text: str
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = self.encode_chunk(chunk)
            ids.extend(chunk_ids)

        return ids

    def decode(self, ids: list):
        """
        Returns the real string codified in ids.
        
        :param ids: List of integers.
        :type ids: list
        """
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text