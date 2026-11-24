import pytest

torch = pytest.importorskip("torch")

from flow3d.contracts import Flow3DConditions
from flow3d.models.conditions import ConditionEncoder


def encode(encoder, conditions, history, stats, *, origin=2.0, target=3.0):
    size = len(conditions)
    return encoder(conditions, history, stats, origin_times_s=[origin] * size, target_times_s=[target] * size)


def test_matched_permuted_constant_missing_and_regime_history():
    torch.manual_seed(2)
    encoder = ConditionEncoder(dim=16)
    z, b = torch.randn(2, 4, 5, 3), torch.randn(2, 4, 6)
    cond = [{"text_prompt": f"3D cylinder wake Re={re}", "reynolds_number": re,
             "u0_m_s": speed, "dt_phys_s": dt, "dt_ctu": 0.1}
            for re, speed, dt in ((40_000, 0.2, 0.15), (500_000, 2.0, 0.015))]
    tokens = encode(encoder, cond, z, b)
    assert tokens.shape == (2, 5, 16)
    shuffled = encode(encoder, list(reversed(cond)), z, b)
    torch.testing.assert_close(shuffled[:, :4], tokens.flip(0)[:, :4])
    torch.testing.assert_close(shuffled[:, 4], tokens[:, 4])
    for slot in range(3):
        assert not torch.equal(tokens[0, slot], tokens[1, slot])
    fixed = encode(encoder, [cond[0], cond[0]], z, b)
    torch.testing.assert_close(fixed[0, :4], fixed[1, :4])
    assert not torch.equal(fixed[0, 4], fixed[1, 4])
    changed_history = encode(encoder, cond, z + 1, b)
    torch.testing.assert_close(changed_history[:, :4], tokens[:, :4])
    assert not torch.equal(changed_history[:, 4], tokens[:, 4])
    missing = encode(encoder, [{}, {}], z, b)
    masked = [{"text_prompt": "future vortex phase=5", "reynolds_number": 5000,
               "known_mask": {"reynolds_number": False}}] * 2
    torch.testing.assert_close(encode(encoder, masked, z, b), missing)
    tokens.sum().backward()
    assert all(parameter.grad is not None for parameter in encoder.parameters())


def test_action_value_identity_units_and_issue_time():
    torch.manual_seed(12)
    encoder = ConditionEncoder(dim=10)
    z, b = torch.randn(1, 1, 2, 2, 2, 1), torch.randn(1, 1, 6)
    meta = {"exogenous": True, "action_channel": "inlet", "action_unit": "m/s", "action_times_s": [2., 3., 4.],
            "action_known_at_time_s": 2.}
    action = {"known_mask": {"action_schedule": True}, "valid_mask": {"action_schedule": True},
              "action_schedule": [0., 0., 1.], "extra_metadata": meta}

    def token(row, **kwargs):
        return encode(encoder, [row], z, b, **kwargs)[0, 3]

    assert not torch.equal(token({}), token(action))
    for declaration in ({k: v for k, v in meta.items() if k != "exogenous"}, {**meta, "exogenous": False}):
        torch.testing.assert_close(token({**action, "extra_metadata": declaration}), token({}))
    assert not torch.equal(token(action, target=3), token(action, target=4))
    assert not torch.equal(token(action), token({**action, "extra_metadata": {**meta, "action_unit": "Pa"}}))
    assert not torch.equal(token(action), token({**action, "extra_metadata": {**meta, "action_channel": "outlet"}}))
    late = {**action, "extra_metadata": {**meta, "action_known_at_time_s": 2.5}}
    torch.testing.assert_close(token(late, target=4), token({}))
    torch.testing.assert_close(token(action, target=5), token({}))
    torch.testing.assert_close(token({**action, "valid_mask": {"action_schedule": False}}), token({}))
    torch.testing.assert_close(token(action), token({**action, "action_schedule": [0., 0., -6.]}))
    with pytest.raises(ValueError, match="future truth"):
        token({**action, "extra_metadata": {**meta, "future_b": [3]}})


def test_unspecified_ctu_cadence_remains_missing_without_reference_speed():
    encoder = ConditionEncoder(dim=8)
    z, b = torch.ones(1, 2, 1, 1), torch.ones(1, 2, 6)
    condition = Flow3DConditions(dt_phys_s=0.1)
    typed = encode(encoder, [condition], z, b)
    raw = condition.to_dict()
    assert raw["dt_ctu"] is None
    torch.testing.assert_close(encode(encoder, [raw], z, b), typed)
    with pytest.raises(ValueError, match="dt_ctu"):
        encode(encoder, [{**raw, "dt_ctu": -0.1}], z, b)
    derived = Flow3DConditions(dt_phys_s=0.1, u0_m_s=-2., diameter_m=0.5)
    assert derived.resolved_dt_ctu == pytest.approx(0.4)
    without_explicit_ctu = {**derived.to_dict(), "dt_ctu": None}
    torch.testing.assert_close(encode(encoder, [derived], z, b), encode(encoder, [without_explicit_ctu], z, b))
    masked_speed = Flow3DConditions(dt_phys_s=0.1, u0_m_s=2., known_mask={"u0_m_s": False})
    assert masked_speed.to_dict()["dt_ctu"] is None
    assert Flow3DConditions(dt_phys_s=0.1, dt_ctu=0.7).to_dict()["dt_ctu"] == 0.7
