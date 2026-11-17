import numpy as np

from stage1.data import NpzFlowDataset


def test_npz_dataset_manifest_and_normalized_sample(tmp_path):
    path = tmp_path / "flow.npz"
    frames = np.zeros((2, 6, 3, 4, 5), dtype=np.float32)
    frames[:, 4, 1:, 1:-1, 1:-1] = 1
    frames[:, 0] = 2
    np.savez(path, frames=frames)
    dataset = NpzFlowDataset(path)
    assert len(dataset) == 2
    normalized, stats = dataset.normalized(0)
    assert normalized.shape == (6, 3, 4, 5)
    assert stats.mean_u == 2
    assert dataset.manifest.sha256
