import torch

def gaussian_kernel(x, y, sigma=1.0):
    """
    Compute the Gaussian kernel between two sets of samples.

    Args:
        x (torch.Tensor): Samples from distribution P. Shape: (N, D)
        y (torch.Tensor): Samples from distribution Q. Shape: (M, D)
        sigma (float): Bandwidth for the Gaussian kernel.

    Returns:
        torch.Tensor: Kernel matrix. Shape: (N, M)
    """
    x = x.unsqueeze(1)  # Shape: (N, 1, D)
    y = y.unsqueeze(0)  # Shape: (1, M, D)
    pairwise_diff = x - y
    distances = torch.sum(pairwise_diff ** 2, dim=-1)  # Shape: (N, M)
    return torch.exp(-distances / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions.

    Args:
        x (torch.Tensor): Samples from distribution P. Shape: (N, D)
        y (torch.Tensor): Samples from distribution Q. Shape: (M, D)
        sigma (float): Bandwidth for the Gaussian kernel.

    Returns:
        float: MMD value.
    """
    xx_kernel = gaussian_kernel(x, x, sigma)  # Kernel for P, P
    yy_kernel = gaussian_kernel(y, y, sigma)  # Kernel for Q, Q
    xy_kernel = gaussian_kernel(x, y, sigma)  # Kernel for P, Q

    mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
    return mmd.item()


if __name__ == "__main__":
    import numpy as np

    x = torch.randn(100, 10)
    y = torch.randn(100, 10)

    mmd_value = compute_mmd(x, y, sigma=1.0)
    print(f"MMD Value: {mmd_value:.3f}")

    x = torch.randn(100, 10)
    y = torch.randn(100, 10) + 1.0

    mmd_value = compute_mmd(x, y, sigma=2.0)
    print(f"MMD Value: {mmd_value:.3f}")

    x = torch.randn(100, 10)
    y = torch.randn(100, 10) * 0.5
    mmd_value = compute_mmd(x, y, sigma=1.0)
    print(f"MMD Value: {mmd_value:.3f}")

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    mmd_value = compute_mmd(x_centered, y_centered, sigma=1.0)
    print(f"MMD Value (variance-only): {mmd_value:.3f}")
