import numpy as np


def one_hot_encode(seqs, length, num_tokens):
    data = np.zeros(
        (len(seqs), length, num_tokens),
        dtype='float32'
    )
    for i, seq in enumerate(seqs):
        for t, token in enumerate(seq):
            data[i, t, token] = 1.
    return data
