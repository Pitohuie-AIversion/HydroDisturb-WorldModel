import pytest

torch = pytest.importorskip("torch")
from flow3d.models.dynamics import CleanQueryModeRes


@pytest.fixture
def example():
    with torch.random.fork_rng():
        torch.manual_seed(137)
        model = CleanQueryModeRes(channels=8, history_length=4, width=12, num_heads=3)
        z = torch.randn(2, 4, 8, 2, 2, 3)
        b = torch.randn(2, 4, 6)
        b[..., 4:] = b[..., 4:].exp()
        conditions = torch.randn(2, 5, 256)
        dt = torch.tensor([0.125, 0.5])
    return model, z, b, conditions, dt


def test_initial_step_matches_delta_and_persistence_at_all_history_lengths(example):
    model, z, b, conditions, dt = example
    for length in (1, 2, 4):
        pred, stats = model(z[:, -length:], b[:, -length:], conditions, dt)
        delta = z[:, -1] - z[:, -min(length, 2)]
        torch.testing.assert_close(pred, z[:, -1] + delta)
        torch.testing.assert_close(stats, b[:, -1])


def test_clean_track_is_prefix_causal_and_query_reads_all_five_slots(example):
    model, z, b, conditions, dt = example
    # Open the initially zero output projection so the query's sensitivities are observable.
    with torch.no_grad():
        model.latent_rate[-1].weight[:8, :8] = torch.eye(8)
        model.background_rate[-1].bias.copy_(torch.tensor([1., 2., 3., 4., -2., 1.]))
    changed = z.clone()
    changed[:, 2:] += 15.0
    torch.testing.assert_close(model.clean_tokens(changed)[:, :2], model.clean_tokens(z)[:, :2])
    conditions.requires_grad_()
    pred, stats = model(z, b, conditions, dt)
    pred.square().sum().backward()
    assert conditions.grad.flatten(2).norm(dim=-1).min() > 0
    assert model.temporal_qkv.weight.grad.norm() > 0
    assert model.spatial.qkv.weight.grad.norm() > 0
    other, other_stats = model(z, b, -conditions, dt)
    assert not torch.allclose(other, pred)
    torch.testing.assert_close(other_stats, stats)
    assert torch.all(stats[:, 4:] > 0)


def test_time_scaling_and_background_unit_equivariance(example):
    model, z, b, conditions, dt = example
    with torch.no_grad():
        model.latent_rate[-1].bias.fill_(0.5)
        model.background_rate[-1].weight.fill_(0.01)
        model.background_rate[-1].bias.fill_(0.4)
    z1, b1 = model(z, b, conditions, dt)
    z2, _ = model(z, b, conditions, dt * 2)
    torch.testing.assert_close(z2 - z1, torch.full_like(z1, 0.5) * dt.view(-1, 1, 1, 1, 1))
    scale = torch.tensor([2., 2., 2., 10., 2., 10.])
    offset = torch.tensor([1., 3., -4., 129., 0., 0.])
    _, transformed = model(z, b * scale + offset, conditions, dt)
    torch.testing.assert_close(transformed, b1 * scale + offset)


def test_rejects_unknown_slots_invalid_cadence_and_scales(example):
    model, z, b, conditions, dt = example
    with pytest.raises(ValueError, match="five"):
        model(z, b, conditions[:, :4], dt)
    with pytest.raises(ValueError, match="dt_ctu"):
        model(z, b, conditions, dt * 0)
    invalid_b = b.clone()
    invalid_b[0, 0, 5] = -2.0
    with pytest.raises(ValueError, match="positive scales"):
        model(z, invalid_b, conditions, dt)


def test_softplus_uses_si_floors_for_tiny_scales_and_stable_anchor_for_large_pressure(example):
    _, z, b, conditions, dt = example
    model = CleanQueryModeRes(8, 4, 12, 3, scale_floor_uvw=2e-6, scale_floor_p=1e-4)
    b[..., 4:] = torch.tensor([2e-36, 3e24])
    b[1, :, 5] = model.scale_floors[1]
    _, stats = model(z, b, conditions, dt)
    torch.testing.assert_close(stats[0, 5], b[0, -1, 5])
    torch.testing.assert_close(stats[:, 4], model.scale_floors[0].expand(2), rtol=2e-7, atol=1e-13)
    torch.testing.assert_close(stats[1, 5], model.scale_floors[1], rtol=2e-7, atol=1e-12)
    with torch.no_grad():
        model.background_rate[-1].bias.fill_(-40)
    _, reduced = model(z, b, conditions, dt)
    assert torch.all(torch.isfinite(reduced))
    assert torch.all(reduced[:, 4:] >= model.scale_floors)
    assert 0 < reduced[0, 5] < stats[0, 5]
