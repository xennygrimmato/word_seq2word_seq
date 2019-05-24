import numpy as np
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def one_hot_encode(seqs, length, num_tokens, padding='post'):
    one_hot_seqs = [to_categorical(seq, num_classes=num_tokens) for seq in seqs]
    return pad_sequences(one_hot_seqs, maxlen=length, dtype='float32', padding=padding, value=np.zeros(shape=(num_tokens,)))


def add_n(seqs, length, n, padding='post'):
    new_seqs = [[e+n for e in seq] for seq in seqs]
    return pad_sequences(new_seqs, maxlen=length, padding=padding)
