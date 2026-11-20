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
    # Layout is (B, 3, Z, Y, X) with velocity channels (u, v, w) along (X, Y, Z)
    # and spacing = (dz, dy, dx).
    m = mask.ge(0.5).to(velocity.dtype)
    valid_z = m.clone(); valid_z[:, :, 0] = 0; valid_z[:, :, -1] = 0
    valid_y = m.clone(); valid_y[:, :, :, 0] = 0; valid_y[:, :, :, -1] = 0
    valid_x = m.clone(); valid_x[:, :, :, :, 0] = 0; valid_x[:, :, :, :, -1] = 0
    valid_z = valid_z * torch.roll(m, 1, -3) * torch.roll(m, -1, -3)
    valid_y = valid_y * torch.roll(m, 1, -2) * torch.roll(m, -1, -2)
    valid_x = valid_x * torch.roll(m, 1, -1) * torch.roll(m, -1, -1)
    return (
        diff(velocity[:, 0:1], -1, spacing[2]) * valid_x
        + diff(velocity[:, 1:2], -2, spacing[1]) * valid_y
        + diff(velocity[:, 2:3], -3, spacing[0]) * valid_z
    )
