import torch

def inverse_2x2(m) -> torch.Tensor | None:
    """
    Compute the inverse of a 2x2 matrix using PyTorch.
    
    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    matrix = torch.as_tensor(m, dtype=torch.float)
    a, b, c, d  = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

    # inv = adj(A)/det(A)
    det = a*d - b*c

    if torch.abs(det) < 1e-9:
        return None
    
    inv = torch.tensor([[d,-b],[-c,a]])/det

    return inv