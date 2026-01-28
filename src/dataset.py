from pathlib import Path
from tokenizer import Tokenizer
import numpy as np
import argparse
import os

# https://www.gutenberg.org/ebooks/author/505

# El Quijote + Novelas ejemplares
#t.train(data, 1024) training tokens: 1_441_293 validation tokens: 161_073

def train_tokenizer(dataset: Path, vocab_size: int):
    """
    Docstring para train_tokenizer
    
    :param dataset: Descripción
    :type dataset: Path
    :param vocab_size: Descripción
    :type vocab_size: int
    """
    with open(dataset, 'r', encoding='utf-8') as f:
        data = f.read()

    t = Tokenizer()
    t.train(data, vocab_size)
    t.save("tokenizer_models/" + dataset.stem + str(vocab_size))

def dataset_to_numpy(dataset: Path, tokenizer_model: Path, train_split: float=0.9):
    """
    Docstring para dataset_to_numpy
    
    :param dataset: Descripción
    :type dataset: Path
    :param train: Descripción
    :type train: float
    """
    assert 0 < train_split < 1
    with open(dataset, 'r', encoding='utf-8') as f:
        data = f.read()

    n = len(data)
    train_data = data[:int(n*train_split)]
    val_data = data[int(n*train_split):]

    t = Tokenizer()
    t.load(tokenizer_model)
    train_ids = t.encode(train_data)
    val_ids = t.encode(val_data)

    print("Number of training tokens:", len(train_ids))
    print("Number of validation tokens:", len(val_ids))

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile("train" + tokenizer_model.stem + ".bin")
    val_ids.tofile("val" + tokenizer_model.stem + ".bin")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="whether to train the tokenizer or not", required=False, type=bool, default=False)
    parser.add_argument("-td", "--train_dataset", help="path to dataset used in train", required=False, type=Path)
    parser.add_argument("-vs", "--vocab_size", help="size of the desired vocabulary", required=False, type=int)
    parser.add_argument("-ed", "--encode_dataset", help="path to dataset to infer", required=True, type=Path)
    parser.add_argument("-m", "--model", help="path to a Tokenizer model", required=False, type=Path)
    parser.add_argument("-ts", "--train_split", help="Proportion of training data split", required=False, type=float, default=0.9)

    args = parser.parse_args()

    if not os.path.isfile(args.encode_dataset):
        parser.error(f"Dataset file to encode does not exist or is not a file: {args.encode_dataset}")
    
    if not args.train and not os.path.isfile(args.model):
        parser.error(f"Tokenizer model file to encode does not exist or is not a file: {args.model}")

    if args.train and not os.path.isfile(args.train_dataset):
        parser.error(f"Training dataset file does not exist or is not a file: {args.train_dataset}")
    
    if args.train and not args.vocab_size:
        parser.error(f"In training mode a vocab_size needs to be provided.")

    if args.train:
        train_tokenizer(args.train_dataset, args.vocab_size)

    if args.train and not args.model:
        args.model = Path("tokenizer_models/" + args.train_dataset.stem + str(args.vocab_size) + ".model")
        # alternatively [model.name for model in Path().glob("tokenizer_models/*model")][0]

    dataset_to_numpy(args.encode_dataset, args.model, args.train_split)

if __name__ == "__main__":
    main()
