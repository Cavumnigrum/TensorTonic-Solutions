import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pe = np.zeros((seq_length, d_model), dtype=np.float64)
    pos = np.arange(seq_length, dtype=np.float64).reshape(-1,1)

    i = np.arange(0, d_model, 2, dtype=np.float64)
    div_term = np.exp(i * (-np.log(10000.0)/ d_model)).reshape(1,-1)

    pe[:, 0::2] = np.sin(pos*div_term)
    pe[:, 1::2] = np.cos(pos*div_term)

    return pe