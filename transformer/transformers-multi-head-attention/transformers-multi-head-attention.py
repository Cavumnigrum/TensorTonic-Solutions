import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    batch_size, seq_length, d_model = Q.shape
    d_k = d_model // num_heads
    
    Q_proj = np.dot(Q,W_q)
    K_proj = np.dot(K,W_k)
    V_proj = np.dot(V,W_v)

    Q_h = Q_proj.reshape(batch_size, seq_length, num_heads, d_k).transpose(0,2,1,3)
    K_h = K_proj.reshape(batch_size, seq_length, num_heads, d_k).transpose(0,2,1,3)
    V_h = V_proj.reshape(batch_size, seq_length, num_heads, d_k).transpose(0,2,1,3)


    scores = np.matmul(Q_h, K_h.transpose(0,1,3,2))/np.sqrt(d_k)

    attn_weights = softmax(scores)

    head_out = np.matmul(attn_weights, V_h)
    concat_out = head_out.transpose(0,2,1,3).reshape(batch_size,seq_length, d_model)
    return np.matmul(concat_out, W_o)