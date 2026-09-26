import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    a_t = torch.tensor(a, dtype = torch.float)
    b_t = torch.tensor(b, dtype = torch.float)

    
    if len(a_t[0]) != len(b_t):
        return -1
    else:
        return torch.matmul(a_t, b_t)
