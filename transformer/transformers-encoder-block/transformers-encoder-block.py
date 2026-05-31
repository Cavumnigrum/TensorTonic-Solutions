import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    m = np.mean(x, axis=-1, keepdims=True)
    d = np.var(x, axis=-1, keepdims=True)

    return gamma * (x-m)/np.sqrt(d+eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch_size, seq_length, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = np.dot(Q, W_q)
    K_proj = np.dot(K, W_k)
    V_proj = np.dot(V, W_v)

    Q_h = Q_proj.reshape(batch_size, seq_length, num_heads, d_k ).transpose(0,2,1,3)
    K_h = K_proj.reshape(batch_size, seq_length, num_heads, d_k ).transpose(0,2,1,3)
    V_h = V_proj.reshape(batch_size, seq_length, num_heads, d_k ).transpose(0,2,1,3)

    scores = np.matmul(Q_h, K_h.transpose(0,1,3,2)) / np.sqrt(d_k)
    weights = softmax(scores)

    out = np.matmul(weights, V_h)

    concat_out = out.transpose(0,2,1,3).reshape(batch_size, seq_length, d_model)
    return np.matmul(concat_out, W_o)
    
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    x_ = np.maximum(0, np.dot(x, W1) + b1)
    return np.dot(x_, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x_ = multi_head_attention(x,x,x, W_q, W_k, W_v,  W_o, num_heads)
    x_ = layer_norm(x+x_, gamma1, beta1)
    out = layer_norm(x_+feed_forward(x_, W1, b1, W2, b2), gamma2, beta2)
    return out
    