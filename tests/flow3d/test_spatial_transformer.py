from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flow3d.data.schema import load_yaml_or_json_config
from flow3d.models import (
    DirectFieldHead3D,
    Flow3DStage1Autoencoder,
    LatentPhysicsProbe,
    LatentNorm3D,
    SpatialTransformer3D,
    VectorPotentialCurlHead3D,
)


def test_spatial_transformer_rejects_depth_zero_placeholder():
    with pytest.raises(ValueError, match="depth >= 1"):
        SpatialTransformer3D(dim=256, depth=0, num_heads=8, mlp_dim=1024)
    with pytest.raises(ValueError, match="depth >= 1"):
        SpatialTransformer3D(dim=256, depth=-1, num_heads=8, mlp_dim=1024)


def test_spatial_transformer_non_trivial_spatial_mixing_and_attention_gradients():
    torch.manual_seed(42)
    st = SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024)
    z_a = torch.randn(1, 1, 256, 2, 3, 4)
    z_b = z_a.clone()
    # Perturb ONLY spatial token (0, 0, 0) with a non-constant channel vector (so LayerNorm does not cancel it)
    z_b[0, 0, :, 0, 0, 0] = z_b[0, 0, :, 0, 0, 0] + torch.randn(256) * 5.0

    out_a = st(z_a)
    out_b = st(z_b)
    # Other spatial tokens (e.g. (1, 2, 3)) MUST change via global self-attention mixing
    assert not torch.allclose(out_a[0, 0, :, 1, 2, 3], out_b[0, 0, :, 1, 2, 3], atol=1e-4)

    # Attention parameters (qkv, proj, mlp) must receive non-zero finite gradients
    z_grad = z_a.clone().requires_grad_(True)
    loss = st(z_grad).pow(2).mean()
    loss.backward()
    for name in ("qkv.weight", "proj.weight", "mlp.0.weight", "mlp.3.weight"):
        param = dict(st.blocks[0].named_parameters())[name]
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert float(param.grad.abs().sum().item()) > 0.0
    assert z_grad.grad is not None and torch.isfinite(z_grad.grad).all()


def test_spatial_transformer_3d_pos_encoding_breaks_spatial_permutation_symmetry():
    torch.manual_seed(7)
    st = SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024)
    st.eval()

    z = torch.randn(1, 256, 2, 3, 4)
    z_perm = z.flip(dims=(-3, -2, -1))
    with torch.no_grad():
        out = st(z)
        out_perm = st(z_perm)
    # If 3D positional encoding were missing/invariant, flipping input would equal flipped output
    assert not torch.allclose(out.flip(dims=(-3, -2, -1)), out_perm, atol=1e-5)


def test_spatial_transformer_strictly_zero_temporal_mixing():
    torch.manual_seed(13)
    st = SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024)
    st.eval()

    z_a = torch.randn(2, 2, 256, 2, 3, 4)
    z_b = z_a.clone()
    # Wildly perturb frame t=1 across all spatial locations
    z_b[:, 1] = torch.randn_like(z_b[:, 1]) * 100.0 + 25.0

    with torch.no_grad():
        out_a = st(z_a)
        out_b = st(z_b)

    # Frame t=0 must remain bit-identical
    assert torch.equal(out_a[:, 0], out_b[:, 0])
    # Frame t=1 must differ
    assert not torch.allclose(out_a[:, 1], out_b[:, 1])


def test_terminal_norm_ablation_and_legacy_checkpoint_compatibility():
    cfg = load_yaml_or_json_config(Path(__file__).resolve().parents[2] / "configs/flow3d/stage1.yaml")
    modern = Flow3DStage1Autoencoder.from_config(cfg)
    assert cfg["latent_contract_version"] == "latent.flow3d.v2"
    assert isinstance(modern.spatial_transformer.norm_out, torch.nn.Identity)

    # Hold spatial mixing and positions fixed, then vary only the bottleneck norm.
    z = torch.ones(1, 256, 2, 2, 2)
    z[..., 1] *= 30
    with torch.no_grad():
        tokens = modern.spatial_transformer(z)
        dataset_scaled = LatentNorm3D(256, "dataset_rms")(tokens)
        token_scaled = LatentNorm3D(256, "affine_free_rmsnorm")(tokens)
    raw_rms = dataset_scaled.square().mean(1).sqrt()
    assert raw_rms.max() > 5 * raw_rms.min()
    torch.testing.assert_close(token_scaled.square().mean(1), torch.ones_like(raw_rms), atol=1e-5, rtol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for mode in LatentNorm3D.VALID_MODES:
            assert LatentNorm3D(256, mode)(tokens.bfloat16()).dtype == torch.float32

    cfg["latent_contract_version"] = "latent.flow3d.v1"
    del cfg["spatial_transformer"]["output_norm"]
    legacy = Flow3DStage1Autoencoder.from_config(cfg).spatial_transformer
    assert isinstance(legacy.norm_out, torch.nn.LayerNorm)
    with torch.no_grad():
        legacy.norm_out.weight.mul_(2.5)
        legacy.norm_out.bias.fill_(0.4)
    restored = SpatialTransformer3D(output_norm="legacy_layer_norm")
    restored.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(z), legacy(z))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        modern.spatial_transformer.load_state_dict(legacy.state_dict(), strict=True)


@pytest.mark.parametrize("shape", [(2, 2, 256, 2, 3, 4), (1, 1, 256, 8, 12, 24), (2, 256, 2, 3, 4)])
def test_spatial_transformer_preserves_small_and_full_stage1_shapes(shape: tuple[int, ...]):
    st = SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024)
    st.eval()
    z = torch.randn(*shape)
    with torch.no_grad():
        out = st(z)
    assert out.shape == shape
    assert torch.isfinite(out).all()


def test_stage1_yaml_load_and_latent_physics_probe(monkeypatch: pytest.MonkeyPatch):
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = repo_root / "configs" / "flow3d" / "stage1.yaml"
    cfg = load_yaml_or_json_config(cfg_path)

    # Verify zero-dependency _parse_simple_yaml fallback produces identical dict to pyyaml
    import yaml

    pyyaml_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    monkeypatch.setitem(sys.modules, "yaml", None)
    fallback_cfg = load_yaml_or_json_config(cfg_path)
    assert fallback_cfg == pyyaml_cfg
    assert cfg == pyyaml_cfg

    assert cfg["schema_version"] == "flow3d.stage1.v1"
    assert cfg["grid"]["spacing_zyx"] == [0.0625, 0.0625, 0.0625]
    assert cfg["grid"]["spacing_zyx_m"] == [0.01875, 0.01875, 0.01875]
    assert cfg["spatial_transformer"]["depth"] >= 1
    assert cfg["spatial_transformer"]["temporal_attention"] is False
    assert cfg["decoder"]["no_encoder_skip"] is True
    assert cfg["latent_norm"]["type"] == "group_norm"
    assert cfg["head"]["use_potential_branch"] is False
    assert "coulomb_gauge_weight" not in cfg["head"]
    assert cfg["loss"]["coulomb_gauge_weight"] == 0.01
    assert cfg["loss"]["harmonic_laplacian_weight"] == 0.01
    assert cfg["runtime"]["num_gpus"] == "auto"
    assert cfg["runtime"]["scale_batch_linearly_with_n"] is True
    assert cfg["runtime"]["maximize_all_resources"] is True

    plan = Flow3DStage1Autoencoder.resolve_runtime_plan(cfg, detected_gpus=7, detected_cpus=80)
    assert plan.active_gpus == 7
    assert plan.global_batch_size == 7 * int(cfg["runtime"]["per_gpu_batch_size"])

    # Enable probe via config copy and verify Head-A and Head-B (+ potential branch) build cleanly
    cfg_probe = copy.deepcopy(cfg)
    cfg_probe["probe"]["enabled"] = True
    model_a = Flow3DStage1Autoencoder.from_config(cfg_probe)
    assert model_a.probe is not None
    assert isinstance(model_a.head, DirectFieldHead3D)
    assert model_a.default_spacing_zyx == (0.0625, 0.0625, 0.0625)

    cfg_b = copy.deepcopy(cfg)
    cfg_b["head"]["type"] = "head_b"
    cfg_b["head"]["use_potential_branch"] = True
    model_b = Flow3DStage1Autoencoder.from_config(cfg_b)
    assert isinstance(model_b.head, VectorPotentialCurlHead3D)
    assert model_b.head.use_potential_branch is True
    assert model_b.head.phi_proj is not None

    probe = LatentPhysicsProbe(latent_channels=256, hidden_dim=256, spectrum_bins=16)
    z_clean = torch.randn(2, 2, 256, 2, 3, 4, requires_grad=True)
    p_out = probe(z_clean)
    assert p_out["enstrophy_tilde"].shape == (2, 2, 1)
    assert bool((p_out["enstrophy_tilde"] > 0.0).all())
    assert p_out["log_spectrum_tilde"].shape == (2, 2, 16)
    (p_out["enstrophy_tilde"].mean() + p_out["log_spectrum_tilde"].pow(2).mean()).backward()
    assert z_clean.grad is not None and torch.isfinite(z_clean.grad).all()


@pytest.mark.parametrize(
    "section,key,bad_val,err_match",
    [
        ("spatial_transformer", "temporal_attention", True, "temporal_attention"),
        ("spatial_transformer", "pre_ln", False, "pre_ln"),
        ("spatial_transformer", "use_sdpa", False, "use_sdpa"),
        ("encoder", "use_s2c", False, "use_s2c"),
        ("decoder", "no_encoder_skip", False, "no_encoder_skip"),
        ("decoder", "use_c2s", False, "use_c2s"),
        ("decoder", "geometry_channels", 1, "geometry_channels"),
        ("decoder", "geometry_channels", 3, "geometry_channels"),
    ],
)
def test_from_config_fail_closed_invariants(section: str, key: str, bad_val: object, err_match: str):
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml_or_json_config(repo_root / "configs" / "flow3d" / "stage1.yaml")
    cfg[section][key] = bad_val
    with pytest.raises(ValueError, match=err_match):
        Flow3DStage1Autoencoder.from_config(cfg)


def test_from_config_wires_hard_mask_output():
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml_or_json_config(repo_root / "configs" / "flow3d" / "stage1.yaml")
    cfg["head"]["type"] = "head_b"
    cfg["decoder"]["hard_mask_output"] = False

    model = Flow3DStage1Autoencoder.from_config(cfg)
    assert model.hard_mask_output is False
    assert model.readout.hard_mask_output is False
    assert isinstance(model.head, VectorPotentialCurlHead3D)
    assert model.head.hard_mask_output is False

    # Verify solid cells retain unmasked denormalized offset when hard_mask_output=False
    nf = torch.zeros(1, 4, 8, 8, 8)
    b = torch.tensor([[2.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    mask = torch.ones(1, 1, 8, 8, 8)
    mask[:, :, :2, :2, :2] = 0.0
    out_unmasked = model.readout(nf, b, mask)
    assert torch.all(out_unmasked[:, 0, :2, :2, :2] == 2.0)

    cfg["decoder"]["hard_mask_output"] = True
    model_masked = Flow3DStage1Autoencoder.from_config(cfg)
    assert model_masked.readout.hard_mask_output is True
    out_masked = model_masked.readout(nf, b, mask)
    assert torch.all(out_masked[:, :, :2, :2, :2] == 0.0)
