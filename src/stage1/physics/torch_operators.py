from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def torch_divergence(velocity, mask, spacing=(1.0, 1.0, 1.0)):
    if torch is None:
        raise ImportError("torch_divergence requires PyTorch")
    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape (B,3,D,H,W)")
    if mask.ndim == 4:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        raise ValueError("mask must have one channel")

    def diff(x, axis, h):
        out = torch.zeros_like(x)
        if axis == -3:
            out[:, :, 1:-1] = (x[:, :, 2:] - x[:, :, :-2]) / (2 * h)
        elif axis == -2:
            out[:, :, :, 1:-1] = (x[:, :, :, 2:] - x[:, :, :, :-2]) / (2 * h)
        else:
            out[:, :, :, :, 1:-1] = (x[:, :, :, :, 2:] - x[:, :, :, :, :-2]) / (2 * h)
        return out

    # The simple valid mask uses neighboring fluid cells and zeroes boundaries.
    m = mask.ge(0.5).to(velocity.dtype)
    valid_d = m.clone(); valid_d[:, :, 0] = 0; valid_d[:, :, -1] = 0
    valid_h = m.clone(); valid_h[:, :, :, 0] = 0; valid_h[:, :, :, -1] = 0
    valid_w = m.clone(); valid_w[:, :, :, :, 0] = 0; valid_w[:, :, :, :, -1] = 0
    valid_d = valid_d * torch.roll(m, 1, -3) * torch.roll(m, -1, -3)
    valid_h = valid_h * torch.roll(m, 1, -2) * torch.roll(m, -1, -2)
    valid_w = valid_w * torch.roll(m, 1, -1) * torch.roll(m, -1, -1)
    return diff(velocity[:, 0:1], -3, spacing[0]) * valid_d + diff(velocity[:, 1:2], -2, spacing[1]) * valid_h + diff(velocity[:, 2:3], -1, spacing[2]) * valid_w
