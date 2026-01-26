import pytest
from tokenizer import Tokenizer

train_text = """In Chapter 2, we looked at machine learning models that treat pixels as being inde‐pendent inputs. Traditional fully connected neural network layers perform poorly onimages because they do not take advantage of the fact that adjacent pixels are highlycorrelated (see Figure 3-1). Moreover, fully connecting multiple layers does not makeany special provisions for the 2D hierarchical nature of images. Pixels close to eachother work together to create shapes (such as lines and arcs), and these shapes them‐selves work together to create recognizable parts of an object (such as the stem andpetals of a flower)."""
val_text = """The deep neural network that we developed in Chapter 2 had two hidden layers, onewith 64 nodes and the other with 16 nodes. One way to think about this networkarchitecture is shown in Figure 3-2. In some sense, all the information contained inthe input image is being represented by the penultimate layer, whose output consistsof 16 numbers. These 16 numbers that provide a representation of the image arecalled an embedding. Of course, earlier layers also capture information from the inputimage, but those are typically not used as embeddings because they are missing someof the hierarchical information."""

def test_encode_decode_train():
    tokenizer = Tokenizer()
    tokenizer.train(train_text)
    assert train_text == tokenizer.decode(tokenizer.encode(train_text))

def test_encode_decode_book():
    tokenizer = Tokenizer()
    tokenizer.train(train_text)
    assert val_text == tokenizer.decode(tokenizer.encode(val_text))

if __name__ == "__main__":
    pytest.main()