from __future__ import annotations

import numpy as np

from stage1 import Stage1Autoencoder
from stage1.models import torch_available


def main() -> None:
    if not torch_available():
        print("PyTorch is not installed; run `python -m pip install -e .[torch]` before the 3D smoke test.")
        return
    frame = np.zeros((6, 16, 24, 32), dtype=np.float32)
    frame[:3] = np.random.default_rng(7).normal(size=(3, 16, 24, 32)).astype(np.float32)
    frame[3] = np.random.default_rng(8).normal(size=(16, 24, 32)).astype(np.float32)
    frame[4, 2:-2, 2:-2, 2:-2] = 1.0
    frame[5] = 1.0
    pipe = Stage1Autoencoder()
    output = pipe.forward(frame)
    print({"latent_shape": tuple(output.z_clean.shape), "reconstruction_shape": tuple(output.reconstruction.shape), "contract": output.contract.to_dict()})


if __name__ == "__main__":
    main()
