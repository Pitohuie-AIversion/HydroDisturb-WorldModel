import numpy as np

from stage1.data.normalization import FlowNormalizer, NormalizationStats


def sample_frame():
    frame = np.zeros((6, 3, 4, 5), dtype=np.float32)
    frame[0] = 2.0
    frame[1] = np.arange(60, dtype=np.float32).reshape(3, 4, 5) / 10
    frame[2] = -1.0
    frame[3] = np.linspace(-1, 1, 60, dtype=np.float32).reshape(3, 4, 5)
    frame[4, 1:, 1:-1, 1:-1] = 1.0
    frame[5] = np.linspace(-10, 10, 60, dtype=np.float32).reshape(3, 4, 5)
    return frame


def test_normalization_is_reversible_and_clips_geometry():
    normalizer = FlowNormalizer(eps=1e-6, sdf_clip=8.0)
    frame = sample_frame()
    normalized, stats = normalizer.normalize(frame)
    restored = normalizer.denormalize(normalized, stats)
    assert np.array_equal(normalized[4], (frame[4] > 0.5).astype(np.float32))
    assert normalized[5].min() >= -1.0 and normalized[5].max() <= 1.0
    assert np.allclose(restored[:4], frame[:4], atol=1e-5)
    assert np.array_equal(restored[4], frame[4])
    assert np.allclose(restored[5], np.clip(frame[5], -8, 8), atol=1e-5)


def test_joint_velocity_scale_is_not_divided_by_three():
    frame = np.zeros((6, 2, 2, 2), dtype=np.float32)
    frame[4] = 1
    frame[0, 0] = 1
    frame[0, 1] = -1
    frame[1, :, 0] = 1
    frame[1, :, 1] = -1
    stats = FlowNormalizer().fit(frame)
    normalized, _ = FlowNormalizer().normalize(frame, stats)
    energy = 0.5 * np.mean(normalized[0] ** 2 + normalized[1] ** 2 + normalized[2] ** 2)
    assert np.isclose(energy, 0.5, atol=1e-5)


def test_stats_require_six_values():
    try:
        NormalizationStats.from_array(np.zeros(5))
    except ValueError:
        pass
    else:
        raise AssertionError("expected a six-value contract")


def test_batch_normalization_keeps_per_frame_statistics():
    normalizer = FlowNormalizer()
    first = sample_frame()
    second = sample_frame().copy()
    second[0] += 100
    normalized, stats = normalizer.normalize_batch(np.stack([first, second]))
    assert len(stats) == 2
    assert stats[0].mean_u != stats[1].mean_u
    restored = normalizer.denormalize_batch(normalized, stats)
    expected = np.stack([first, second]).copy()
    expected[:, 5] = np.clip(expected[:, 5], -8, 8)
    expected[:, 4] = (expected[:, 4] > 0.5).astype(np.float32)
    assert np.allclose(restored, expected, atol=1e-5)


def test_solid_nan_does_not_poison_stats_and_fluid_nan_raises():
    normalizer = FlowNormalizer(eps=1e-6)
    frame = sample_frame()
    # Inject NaNs inside solid cells (mask == 0)
    solid = frame[4] <= 0.5
    frame[0, solid] = np.nan
    frame[3, solid] = np.nan
    normalized, stats = normalizer.normalize(frame)
    assert np.all(np.isfinite(stats.as_array()))
    assert np.all(np.isfinite(normalized))

    # Inject NaN inside a fluid cell (mask == 1) -> must fail closed
    bad = sample_frame()
    bad[0, 1, 1, 1] = np.nan
    try:
        normalizer.fit(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when fluid cell contains NaN")


def test_constant_field_applies_scale_floor_in_fit():
    normalizer = FlowNormalizer(eps=1e-5)
    frame = np.zeros((6, 3, 4, 5), dtype=np.float32)
    frame[0] = 0.5
    frame[3] = 10.0
    frame[4] = 1.0
    stats = normalizer.fit(frame)
    assert stats.velocity_scale == 1e-5
    assert stats.pressure_scale == 1e-5
