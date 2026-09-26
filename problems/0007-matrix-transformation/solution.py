import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    inv_T, info_T = torch.linalg.inv_ex(T_t, check_errors=False)
    inv_S, info_S = torch.linalg.inv_ex(S_t, check_errors=False)

    if (info_T != 0 and info_S != 0):
        return -1

    else:
        return torch.matmul(inv_T, torch.matmul(A_t, S_t))
