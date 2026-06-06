import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    batch, seq_length, d_k = Q.size()
    QK = torch.matmul(Q, K.transpose(-2,-1))
    scores = QK/math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    return torch.matmul(scores,V)
    