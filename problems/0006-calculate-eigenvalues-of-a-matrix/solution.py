import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    egvals, egvectors = torch.linalg.eig(matrix)
    if torch.all(egvals.imag == 0):
        _, indices = torch.sort(egvals.real, descending=True)
    else:
        _, indices = torch.sort(torch.abs(egvals), descending=True)

    return egvals[indices]