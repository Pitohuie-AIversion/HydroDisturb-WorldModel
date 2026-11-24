from __future__ import annotations

import inspect

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flow3d.contracts import PhysicsContractError
from flow3d.data.fixtures import make_analytical_divergence_free_field_3d, make_cylinder_geometry_3d
from flow3d.data.normalization import Flow3DNormalizer
from flow3d.models import (
    C2S3d,
    DCAE3DDecoder,
    DCAE3DEncoder,
    DirectFieldHead3D,
    ElasticRuntimePlan,
    Flow3DStage1Autoencoder,
    LatentNorm3D,
    LatentPhysicsProbe,
    PhysicalReadout3D,
    S2C3d,
    SpatialTransformer3D,
    VectorPotentialCurlHead3D,
    configure_elastic_hardware,
    resolve_elastic_runtime_plan,
)
from flow3d.physics.boundary import first_fluid_shell_mask
from flow3d.physics.operators import curl_h, div_h, valid_div_of_curl_mask
from flow3d.physics.torch_operators import (
    _broadcast_mask_to_spatial,
    torch_build_full_3d_stencil_mask,
    torch_curl_h,
    torch_div_h,
    torch_helmholtz_project,
    torch_valid_div_of_curl_mask,
)


def _make_small_batch(
    bsz: int = 2,
    t_len: int = 2,
    shape_zyx: tuple[int, int, int] = (16, 24, 32),
    spacing: tuple[float, float, float] = (0.01875, 0.01875, 0.01875),
    seed: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    geom_np, _ = make_cylinder_geometry_3d(shape_zyx, spacing, cylinder_diameter_m=0.15)
    base_field, _ = make_analytical_divergence_free_field_3d(shape_zyx, spacing, u0_m_s=0.5)
    fluid_mask_np = geom_np[0] > 0.5

    fields_np = np.zeros((bsz, t_len, 4, *shape_zyx), dtype=np.float32)
    for b_idx in range(bsz):
        for t_idx in range(t_len):
            noise = rng.normal(scale=0.02, size=(4, *shape_zyx)).astype(np.float32)
            f = base_field + noise
            f[:, ~fluid_mask_np] = 0.0
            fields_np[b_idx, t_idx] = f

    normalizer = Flow3DNormalizer()
    norm_np, b_np = normalizer.normalize(fields_np, geom_np[0])
    geom_batch_np = np.broadcast_to(geom_np[None, ...], (bsz, 2, *shape_zyx)).copy()

    return (
        torch.from_numpy(fields_np),
        torch.from_numpy(norm_np),
        torch.from_numpy(geom_batch_np),
        torch.from_numpy(b_np),
    )


def test_torch_vs_numpy_div_and_curl_parity_and_analytical_div_free():
    shape_zyx = (16, 24, 32)
    spacing = (0.01875, 0.01875, 0.01875)
    geom_np, _ = make_cylinder_geometry_3d(shape_zyx, spacing, cylinder_diameter_m=0.15)
    mask_np = geom_np[0]
    field_np, a_np = make_analytical_divergence_free_field_3d(shape_zyx, spacing, u0_m_s=0.5)

    # Rescale A so curl_h(A) & div_h values match unit-scale FP32 parity cleanly
    a_t = torch.from_numpy(a_np).unsqueeze(0).requires_grad_(True)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0)

    curl_np = curl_h(a_np, spacing, mask=mask_np)
    curl_t = torch_curl_h(a_t, spacing, mask=mask_t)
    assert np.allclose(curl_t[0].detach().numpy(), curl_np, atol=1e-5, rtol=1e-5)

    div_np = div_h(curl_np[0], curl_np[1], curl_np[2], spacing, mask=mask_np)
    div_t = torch_div_h(curl_t[:, 0], curl_t[:, 1], curl_t[:, 2], spacing, mask=mask_t)
    assert np.allclose(div_t[0].detach().numpy(), div_np, atol=1e-5, rtol=1e-5)

    interior_np = valid_div_of_curl_mask(mask_np) > 0.5
    interior_t = torch_valid_div_of_curl_mask(mask_t) > 0.5
    assert np.array_equal(interior_np, interior_t[0].numpy())
    assert float(div_t[0][interior_t[0]].abs().max().item()) < 1e-5

    # Boundary-band truncation check on analytical field velocity + curl
    div_u_t = torch_div_h(
        torch.from_numpy(field_np[0:1]),
        torch.from_numpy(field_np[1:2]),
        torch.from_numpy(field_np[2:3]),
        spacing,
        mask=mask_t,
    )
    assert float(div_u_t[0][interior_t[0]].abs().max().item()) < 1e-5

    # Verify gradient backpropagates cleanly through torch_curl_h & torch_div_h
    loss = div_t.pow(2).sum() + curl_t.pow(2).mean()
    loss.backward()
    assert a_t.grad is not None and torch.isfinite(a_t.grad).all()
    assert float(a_t.grad.abs().sum().item()) > 0.0


def test_s2c_and_c2s_exact_inverse_and_divisibility_validation():
    s2c = S2C3d(factor=2)
    c2s = C2S3d(factor=2)
    x = torch.randn(2, 16, 8, 12, 16)
    y = s2c(x)
    assert y.shape == (2, 128, 4, 6, 8)
    x_rec = c2s(y)
    assert torch.equal(x, x_rec)

    with pytest.raises(ValueError, match="divisible"):
        s2c(torch.randn(1, 4, 7, 8, 8))
    with pytest.raises(ValueError, match="divisible"):
        c2s(torch.randn(1, 10, 4, 4, 4))


@pytest.mark.parametrize(
    "head_type,helmholtz_proj,use_potential_branch",
    [
        ("head_a", False, False),
        ("head_a", True, False),
        ("head_b", False, False),
        ("head_b", False, True),
    ],
)
def test_stage1_small_volume_forward_backward_all_heads(
    head_type: str,
    helmholtz_proj: bool,
    use_potential_branch: bool,
):
    spacing = (0.01875, 0.01875, 0.01875)
    _, x_norm, geom, b_stats = _make_small_batch(bsz=2, t_len=2, shape_zyx=(16, 24, 32), spacing=spacing)

    encoder = DCAE3DEncoder()
    transformer = SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024)
    latent_norm = LatentNorm3D(channels=256, mode="group_norm")
    decoder = DCAE3DDecoder()
    head = (
        DirectFieldHead3D(in_channels=32, helmholtz_projection=helmholtz_proj, helmholtz_iters=10)
        if head_type == "head_a"
        else VectorPotentialCurlHead3D(
            in_channels=32,
            separate_mean_potential=True,
            use_potential_branch=use_potential_branch,
        )
    )
    readout = PhysicalReadout3D()
    probe = LatentPhysicsProbe(latent_channels=256, hidden_dim=128, spectrum_bins=16)

    model = Flow3DStage1Autoencoder(
        encoder=encoder,
        spatial_transformer=transformer,
        latent_norm=latent_norm,
        decoder=decoder,
        head=head,
        readout=readout,
        probe=probe,
        default_spacing_zyx=spacing,
    )

    out = model(x_norm, geom, b=b_stats, run_probe=True)
    assert out["z_clean"].shape == (2, 2, 256, 2, 3, 4)
    assert out["norm_fields"].shape == (2, 2, 4, 16, 24, 32)
    assert out["phys_fields"] is not None and out["phys_fields"].shape == (2, 2, 4, 16, 24, 32)
    assert out["probe"] is not None
    assert out["probe"]["enstrophy_tilde"].shape == (2, 2, 1)
    assert out["probe"]["log_spectrum_tilde"].shape == (2, 2, 16)

    loss = (
        out["phys_fields"].pow(2).mean()
        + out["head_aux"]["div_residual"].pow(2).mean()
        + out["probe"]["enstrophy_tilde"].mean()
        + out["probe"]["log_spectrum_tilde"].pow(2).mean()
    )
    if head_type == "head_b":
        assert "vector_potential" in out["head_aux"]
        assert "coulomb_gauge_loss" in out["head_aux"]
        assert "harmonic_laplacian_loss" in out["head_aux"]
        assert "potential_phi" in out["head_aux"]
        if use_potential_branch:
            assert out["head_aux"]["potential_phi"] is not None
            assert out["head_aux"]["potential_phi"].shape == (2, 2, 16, 24, 32)
        else:
            assert out["head_aux"]["potential_phi"] is None
            assert float(out["head_aux"]["harmonic_laplacian_loss"].item()) == 0.0
        loss = (
            loss
            + 0.01 * out["head_aux"]["coulomb_gauge_loss"]
            + 0.01 * out["head_aux"]["harmonic_laplacian_loss"]
        )

    loss.backward()

    # Check finite non-zero gradients across every active submodule (including latent_norm)
    for name, module in (
        ("encoder", encoder),
        ("spatial_transformer", transformer),
        ("latent_norm", latent_norm),
        ("decoder", decoder),
        ("head", head),
        ("probe", probe),
    ):
        grads = [p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0, f"{name} received no gradients"
        assert all(torch.isfinite(g).all() for g in grads), f"{name} has non-finite gradients"
        assert sum(float(g.abs().sum().item()) for g in grads) > 0.0, f"{name} gradients are all zero"


def test_zero_encoder_decoder_skip_isolation_and_geometry_sensitivity():
    # Signature check: Decoder accepts strictly (z_clean, geometry)
    sig = inspect.signature(DCAE3DDecoder.forward)
    assert list(sig.parameters.keys()) == ["self", "z_clean", "geometry"]

    spacing = (0.01875, 0.01875, 0.01875)
    _, x_norm, geom, b_stats = _make_small_batch(bsz=1, t_len=1, shape_zyx=(16, 24, 32), spacing=spacing)

    model = Flow3DStage1Autoencoder(
        encoder=DCAE3DEncoder(),
        spatial_transformer=SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024),
        latent_norm=LatentNorm3D(channels=256, mode="group_norm"),
        decoder=DCAE3DDecoder(),
        head=DirectFieldHead3D(in_channels=32),
        readout=PhysicalReadout3D(),
        default_spacing_zyx=spacing,
    )
    model.eval()

    with torch.no_grad():
        z_clean = model.encode(x_norm, geom)
        dec1 = model.decode(z_clean, geom, b=b_stats)

        # Mutate input field wildly AND remove encoder reference to prove zero hidden skip dependency
        _ = model.encode(x_norm * 100.0 + 7.5, geom)
        del model.encoder
        dec2 = model.decode(z_clean, geom, b=b_stats)
        assert torch.equal(dec1["norm_fields"], dec2["norm_fields"])
        assert torch.equal(dec1["phys_fields"], dec2["phys_fields"])

        # Changing 2-channel geometry with fixed z_clean must change decoded output
        geom_shifted = geom.clone()
        geom_shifted[:, 1] = geom_shifted[:, 1] + 0.5
        dec_geom_diff = model.decode(z_clean, geom_shifted, b=b_stats)
        assert not torch.allclose(dec1["norm_fields"], dec_geom_diff["norm_fields"])


def test_physical_readout_exact_normalizer_inverse_nan_safety_and_scale_guard():
    shape_zyx = (16, 24, 32)
    spacing = (0.01875, 0.01875, 0.01875)
    fields, x_norm, geom, b_stats = _make_small_batch(bsz=2, t_len=2, shape_zyx=shape_zyx, spacing=spacing)
    # Scale up pressure & velocity so large offsets/scales are exercised
    fields[:, :, :3] = fields[:, :, :3] * 3.5 + 1.2
    fields[:, :, 3] = fields[:, :, 3] * 1200.0 - 450.0
    fluid_mask_np = geom[0, 0].numpy()

    normalizer = Flow3DNormalizer()
    norm_np, b_np = normalizer.normalize(fields.numpy(), fluid_mask_np)

    readout = PhysicalReadout3D()
    norm_t = torch.from_numpy(norm_np)
    b_t = torch.from_numpy(b_np)
    mask_t = geom[:, 0:1]  # 5D [B, 1, Z, Y, X] broadcast over T=2

    rec_t = readout(norm_t, b_t, mask_t)
    mask_bool = mask_t.unsqueeze(1).expand(2, 2, 1, *shape_zyx) > 0.5
    # Verify exact inverse inside fluid domain and zero inside solid domain
    assert torch.allclose(rec_t[mask_bool.expand_as(rec_t)], fields[mask_bool.expand_as(fields)], atol=1e-4, rtol=1e-4)
    assert torch.all(rec_t[(~mask_bool).expand_as(rec_t)] == 0.0)

    # NaN in solid cells MUST NOT leak through torch.where
    norm_with_solid_nan = norm_t.clone()
    norm_with_solid_nan[(~mask_bool).expand_as(norm_with_solid_nan)] = float("nan")
    rec_nan_safe = readout(norm_with_solid_nan, b_t, mask_t)
    assert torch.isfinite(rec_nan_safe).all()
    assert torch.allclose(rec_nan_safe, rec_t)

    # NaN in valid fluid domain must fail closed
    norm_with_fluid_nan = norm_t.clone()
    fluid_idx = mask_bool.nonzero(as_tuple=False)[0]
    norm_with_fluid_nan[fluid_idx[0], fluid_idx[1], 0, fluid_idx[3], fluid_idx[4], fluid_idx[5]] = float("nan")
    with pytest.raises(ValueError, match="valid fluid domain"):
        readout(norm_with_fluid_nan, b_t, mask_t)

    # Non-positive scale sigma_uvw or sigma_p must fail closed
    b_bad = b_t.clone()
    b_bad[0, 0, 4] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        readout(norm_t, b_bad, mask_t)
    b_bad2 = b_t.clone()
    b_bad2[0, 0, 5] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        readout(norm_t, b_bad2, mask_t)


def test_helmholtz_projection_and_head_b_operator_compliance():
    shape_zyx = (16, 24, 32)
    spacing = (0.01875, 0.01875, 0.01875)
    geom_np, _ = make_cylinder_geometry_3d(shape_zyx, spacing, cylinder_diameter_m=0.15)
    field_np, a_np = make_analytical_divergence_free_field_3d(shape_zyx, spacing, u0_m_s=0.5)

    mask_t = torch.from_numpy(geom_np[0]).unsqueeze(0)  # [1, Z, Y, X]
    interior_t = torch_valid_div_of_curl_mask(mask_t) > 0.5

    # Head-A Helmholtz projection reduces divergence dramatically on divergent velocity
    rng = torch.Generator().manual_seed(11)
    vel_div = torch.from_numpy(field_np[:3]).unsqueeze(0) + 0.08 * torch.randn((1, 3, *shape_zyx), generator=rng)
    div_before = torch_div_h(vel_div[:, 0], vel_div[:, 1], vel_div[:, 2], spacing, mask=mask_t)
    vel_proj, _ = torch_helmholtz_project(vel_div, spacing, fluid_mask=mask_t, num_iters=25, method="cg")
    div_after = torch_div_h(vel_proj[:, 0], vel_proj[:, 1], vel_proj[:, 2], spacing, mask=mask_t)
    rms_before = float(div_before.pow(2).mean().sqrt().item())
    rms_after = float(div_after.pow(2).mean().sqrt().item())
    assert rms_after < 1e-3 * rms_before

    # Head-B: curl_h(A*) + spatially uniform mean_flow preserves div_h == 0 on valid_div_of_curl_mask
    head_b = VectorPotentialCurlHead3D(in_channels=8, separate_mean_potential=True)
    # Construct decoder feature tensor where a_proj outputs exact analytical A* and mean_head adds U0
    feat = torch.zeros(1, 8, *shape_zyx)
    with torch.no_grad():
        head_b.a_proj.weight.zero_()
        head_b.a_proj.bias.zero_()
        head_b.mean_head.weight.zero_()
        head_b.mean_head.bias.copy_(torch.tensor([0.5, 0.0, 0.0]))
    out_b = head_b(feat, fluid_mask=mask_t, spacing_zyx=spacing)
    assert float(out_b["div_residual"][0][interior_t[0]].abs().max().item()) < 1e-5

    # Non-trivial A* gives exact interior divergence-free velocity via torch_curl_h
    vel_from_a = torch_curl_h(torch.from_numpy(a_np).unsqueeze(0), spacing, mask=mask_t)
    div_a = torch_div_h(vel_from_a[:, 0], vel_from_a[:, 1], vel_from_a[:, 2], spacing, mask=mask_t)
    assert float(div_a[0][interior_t[0]].abs().max().item()) < 1e-5


@pytest.mark.parametrize("mode", ["group_norm", "dataset_rms", "affine_free_rmsnorm"])
def test_latent_norm_ablation_modes_and_invalid_mode(mode: str):
    z = torch.randn(2, 2, 256, 2, 3, 4) * 3.0 + 1.0
    ln = LatentNorm3D(channels=256, mode=mode, num_groups=32, dataset_rms_scale=2.0)
    out = ln(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()

    if mode == "dataset_rms":
        assert torch.allclose(out, z / 2.0)
    elif mode == "affine_free_rmsnorm":
        token_rms = out.pow(2).mean(dim=2).sqrt()
        assert torch.allclose(token_rms, torch.ones_like(token_rms), atol=1e-5)

    with pytest.raises(ValueError, match="unknown latent_norm mode"):
        LatentNorm3D(channels=256, mode="invalid_norm")
    with pytest.raises(ValueError, match="dataset_rms_scale"):
        LatentNorm3D(channels=256, mode="group_norm", dataset_rms_scale=0.0)


def test_active_device_forward_backward_verification():
    # Runs on CUDA when GPU is attached (no CPU skips)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacing = (0.01875, 0.01875, 0.01875)
    _, x_norm, geom, b_stats = _make_small_batch(bsz=1, t_len=1, shape_zyx=(16, 24, 32), spacing=spacing)
    x_norm, geom, b_stats = x_norm.to(device), geom.to(device), b_stats.to(device)

    model = Flow3DStage1Autoencoder(
        encoder=DCAE3DEncoder(),
        spatial_transformer=SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024),
        latent_norm=LatentNorm3D(channels=256, mode="group_norm"),
        decoder=DCAE3DDecoder(),
        head=VectorPotentialCurlHead3D(in_channels=32),
        readout=PhysicalReadout3D(),
        default_spacing_zyx=spacing,
    ).to(device)

    out = model(x_norm, geom, b=b_stats)
    loss = out["phys_fields"].pow(2).mean() + 0.01 * out["head_aux"]["coulomb_gauge_loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert out["phys_fields"].device.type == device.type
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum().item()) for g in grads) > 0.0


@pytest.mark.parametrize("n_gpus", [1, 2, 3, 4, 5, 8, 16])
def test_elastic_n_gpu_runtime_plan_scales_for_any_natural_number_n(n_gpus: int):
    cfg = {
        "runtime": {
            "num_gpus": "auto",
            "strategy": "auto",
            "per_gpu_batch_size": 2,
            "scale_batch_linearly_with_n": True,
            "maximize_all_resources": True,
            "cpu_threads": "auto",
            "dataloader_workers_per_gpu": "auto",
            "pin_memory": True,
            "non_blocking": True,
        }
    }
    plan = resolve_elastic_runtime_plan(cfg, detected_gpus=n_gpus, detected_cpus=80, detected_ram_gib=503.0)
    assert isinstance(plan, ElasticRuntimePlan)
    assert plan.active_gpus == n_gpus
    assert plan.device_ids == tuple(range(n_gpus))
    assert plan.devices == tuple(f"cuda:{i}" for i in range(n_gpus))
    assert plan.primary_device == "cuda:0"
    assert plan.global_batch_size == 2 * n_gpus
    assert plan.cpu_threads_total == 80
    assert plan.threads_per_gpu_worker == max(1, 80 // n_gpus)
    assert plan.total_dataloader_workers == plan.dataloader_workers_per_gpu * n_gpus
    assert plan.strategy == ("single_gpu" if n_gpus == 1 else "data_parallel")
    assert [plan.task_device_for_index(k) for k in range(n_gpus + 2)] == [
        f"cuda:{k % n_gpus}" for k in range(n_gpus + 2)
    ]


def test_elastic_runtime_plan_cpu_fallback_and_validation():
    cpu_plan = resolve_elastic_runtime_plan({"runtime": {"num_gpus": "auto"}}, detected_gpus=0, detected_cpus=16)
    assert cpu_plan.active_gpus == 0
    assert cpu_plan.devices == ("cpu",)
    assert cpu_plan.primary_device == "cpu"
    assert cpu_plan.strategy == "cpu"
    assert cpu_plan.pin_memory is False
    assert cpu_plan.non_blocking is False

    # active_n == 1 with strategy='cpu' must resolve to 'single_gpu' (not 'cpu')
    plan_1_from_cpu_strat = resolve_elastic_runtime_plan(
        {"runtime": {"num_gpus": 1, "strategy": "cpu"}},
        detected_gpus=5,
        detected_cpus=80,
    )
    assert plan_1_from_cpu_strat.active_gpus == 1
    assert plan_1_from_cpu_strat.devices == ("cuda:0",)
    assert plan_1_from_cpu_strat.strategy == "single_gpu"

    # Explicit num_gpus < detected_gpus, silent clamp when strict_gpu_count=False, num_gpus=0, and non-linear batch
    sub_plan = resolve_elastic_runtime_plan({"runtime": {"num_gpus": 2}}, detected_gpus=5, detected_cpus=80)
    assert sub_plan.active_gpus == 2
    assert sub_plan.device_ids == (0, 1)

    clamped_plan = resolve_elastic_runtime_plan(
        {"runtime": {"num_gpus": 8}}, detected_gpus=5, detected_cpus=80, strict_gpu_count=False
    )
    assert clamped_plan.active_gpus == 5

    zero_plan = resolve_elastic_runtime_plan({"runtime": {"num_gpus": 0}}, detected_gpus=5, detected_cpus=80)
    assert zero_plan.active_gpus == 0
    assert zero_plan.strategy == "cpu"

    fixed_bs_plan = resolve_elastic_runtime_plan(
        {"runtime": {"num_gpus": 5, "per_gpu_batch_size": 2, "scale_batch_linearly_with_n": False}},
        detected_gpus=5,
        detected_cpus=80,
    )
    assert fixed_bs_plan.global_batch_size == 2

    with pytest.raises(ValueError, match="num_gpus"):
        resolve_elastic_runtime_plan({"runtime": {"num_gpus": -1}}, detected_gpus=4)
    with pytest.raises(ValueError, match="exceeds available GPUs"):
        resolve_elastic_runtime_plan({"runtime": {"num_gpus": 6}}, detected_gpus=5, strict_gpu_count=True)


def test_forward_elastic_multi_gpu_sharding_parity_and_gradients():
    plan = resolve_elastic_runtime_plan({"runtime": {"num_gpus": "auto"}})
    hw_info = configure_elastic_hardware(plan)
    assert hw_info["torch_num_threads"] >= 1

    spacing = (0.01875, 0.01875, 0.01875)
    # Exercise ALL active GPUs (no artificial cap at 4)
    bsz = max(2, plan.active_gpus) if plan.active_gpus >= 2 else 2
    _, x_norm, geom, b_stats = _make_small_batch(bsz=bsz, t_len=1, shape_zyx=(16, 24, 32), spacing=spacing)

    torch.manual_seed(99)
    model = Flow3DStage1Autoencoder(
        encoder=DCAE3DEncoder(),
        spatial_transformer=SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024),
        latent_norm=LatentNorm3D(channels=256, mode="group_norm"),
        decoder=DCAE3DDecoder(),
        head=VectorPotentialCurlHead3D(in_channels=32),
        readout=PhysicalReadout3D(),
        probe=LatentPhysicsProbe(latent_channels=256, hidden_dim=128, spectrum_bins=16),
        default_spacing_zyx=spacing,
    )

    # Check eval-mode numerical parity between single-device forward and elastic multi-GPU forward
    model.eval()
    primary_dev = torch.device(plan.primary_device)
    model.to(primary_dev)
    with torch.no_grad():
        ref_out = model(
            x_norm.to(primary_dev),
            geom.to(primary_dev),
            b=b_stats.to(primary_dev),
            run_probe=True,
        )
        per_shard_outs = [
            model(
                x_norm[i : i + 1].to(primary_dev),
                geom[i : i + 1].to(primary_dev),
                b=b_stats[i : i + 1].to(primary_dev),
                run_probe=True,
            )
            for i in range(bsz)
        ]
        per_shard_z = torch.cat([o["z_clean"] for o in per_shard_outs], dim=0)
        per_shard_norm = torch.cat([o["norm_fields"] for o in per_shard_outs], dim=0)
        per_shard_phys = torch.cat([o["phys_fields"] for o in per_shard_outs], dim=0)
        elastic_out = model.forward_elastic_multi_gpu(
            x_norm,
            geom,
            b=b_stats,
            run_probe=True,
            num_gpus="auto",
        )
    shard_batch_sizes = elastic_out["elastic_runtime"]["shard_batch_sizes"]
    if plan.active_gpus >= 2:
        assert shard_batch_sizes == [1] * bsz
        # Exact shard-level parity across GPUs 0..(m-1) vs GPU 0
        assert torch.allclose(per_shard_z, elastic_out["z_clean"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(per_shard_norm, elastic_out["norm_fields"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(per_shard_phys, elastic_out["phys_fields"], atol=1e-6, rtol=1e-6)
        # Full-batch B=bsz vs sharded m x (B/m) parity (accounting for 1/(2*dx)=26.7x curl stencil factor)
        assert torch.allclose(ref_out["z_clean"], elastic_out["z_clean"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(ref_out["norm_fields"], elastic_out["norm_fields"], atol=5e-4, rtol=5e-4)
        assert torch.allclose(ref_out["phys_fields"], elastic_out["phys_fields"], atol=5e-4, rtol=5e-4)
        assert torch.allclose(
            ref_out["head_aux"]["coulomb_gauge_loss"],
            elastic_out["head_aux"]["coulomb_gauge_loss"],
            atol=1e-4,
            rtol=1e-4,
        )
    else:
        assert shard_batch_sizes == [bsz]
        # Single-device fallback (n=0 CPU or n=1 GPU): strict B=bsz parity vs ref_out
        assert torch.allclose(ref_out["z_clean"], elastic_out["z_clean"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(ref_out["norm_fields"], elastic_out["norm_fields"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(ref_out["phys_fields"], elastic_out["phys_fields"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            ref_out["head_aux"]["coulomb_gauge_loss"],
            elastic_out["head_aux"]["coulomb_gauge_loss"],
            atol=1e-6,
            rtol=1e-6,
        )
        # B=1 slice parity vs B=bsz forward (accounting for FP32 reduction order & 1/(2*dx) curl factor)
        assert torch.allclose(per_shard_z, elastic_out["z_clean"], atol=5e-4, rtol=1e-4)
        assert torch.allclose(per_shard_norm, elastic_out["norm_fields"], atol=5e-4, rtol=1e-4)
        assert torch.allclose(per_shard_phys, elastic_out["phys_fields"], atol=5e-4, rtol=1e-4)
    expected_used = min(plan.active_gpus, bsz) if plan.active_gpus >= 1 else 0
    assert elastic_out["elastic_runtime"]["active_gpus"] == expected_used
    assert sum(elastic_out["elastic_runtime"]["shard_batch_sizes"]) == bsz

    # Check train-mode backward gradient accumulation across all active GPU shards
    model.train()
    model.zero_grad(set_to_none=True)
    train_out = model.forward_elastic_multi_gpu(x_norm, geom, b=b_stats, run_probe=True, num_gpus="auto")
    loss = (
        train_out["phys_fields"].pow(2).mean()
        + 0.01 * train_out["head_aux"]["coulomb_gauge_loss"]
        + train_out["probe"]["enstrophy_tilde"].mean()
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum().item()) for g in grads) > 0.0


def test_head_b_nonzero_a_proj_no_first_shell_dead_zone_and_fp64_div_free():
    shape_zyx = (16, 24, 32)
    spacing = (0.0625, 0.0625, 0.0625)
    geom_np, _ = make_cylinder_geometry_3d(shape_zyx, spacing, cylinder_diameter_m=0.5)
    _, a_np = make_analytical_divergence_free_field_3d(shape_zyx, spacing, u0_m_s=0.5)
    mask_t = torch.from_numpy(geom_np[0]).unsqueeze(0)  # [1, Z, Y, X]

    head_b = VectorPotentialCurlHead3D(in_channels=3, separate_mean_potential=False)
    with torch.no_grad():
        head_b.a_proj.weight.zero_()
        head_b.a_proj.weight[:, :, 1, 1, 1].copy_(torch.eye(3))
        head_b.a_proj.bias.zero_()
        head_b.p_proj.weight.zero_()
        head_b.p_proj.bias.zero_()

    feat = torch.from_numpy(a_np).unsqueeze(0)  # [1, 3, Z, Y, X]
    out_fp32 = head_b(feat, fluid_mask=mask_t, spacing_zyx=spacing)

    # 1st fluid shell around the cylinder must NOT be zeroed out by torch_curl_h
    shell_mask = torch.from_numpy(first_fluid_shell_mask(geom_np[0]))
    assert bool(shell_mask.any())
    u_shell = out_fp32["norm_fields"][0, :3][:, shell_mask]
    assert float(u_shell.abs().max().item()) > 1e-3
    assert float(u_shell.pow(2).mean().sqrt().item()) > 1e-3

    # Full-stencil fluid cells away from outer box faces (includes 2nd cylinder shell) must have near-zero div
    full_stencil = torch_build_full_3d_stencil_mask(mask_t)[0] > 0.5
    box_interior = torch_valid_div_of_curl_mask(torch.ones_like(mask_t))[0] > 0.5
    valid_full_interior = full_stencil & box_interior
    assert bool(valid_full_interior.any())
    assert float(out_fp32["div_residual"][0][valid_full_interior].abs().max().item()) < 1e-4

    # FP64 exactness check (<= 1e-10)
    head_b_fp64 = copy.deepcopy(head_b).double()
    out_fp64 = head_b_fp64(feat.double(), fluid_mask=mask_t.double(), spacing_zyx=spacing)
    assert float(out_fp64["div_residual"][0][valid_full_interior].abs().max().item()) <= 1e-10


def test_head_b_potential_branch_harmonic_laplacian_loss_and_grad():
    shape_zyx = (16, 24, 32)
    spacing = (0.0625, 0.0625, 0.0625)
    geom_np, _ = make_cylinder_geometry_3d(shape_zyx, spacing, cylinder_diameter_m=0.5)
    mask_t = torch.from_numpy(geom_np[0]).unsqueeze(0)

    # 1. Analytical harmonic vs non-harmonic phi via conv weights + spatial coordinates
    head_b = VectorPotentialCurlHead3D(in_channels=3, separate_mean_potential=True, use_potential_branch=True)
    zz = torch.arange(shape_zyx[0], dtype=torch.float64) * spacing[0]
    yy = torch.arange(shape_zyx[1], dtype=torch.float64) * spacing[1]
    xx = torch.arange(shape_zyx[2], dtype=torch.float64) * spacing[2]
    zg, yg, xg = torch.meshgrid(zz, yy, xx, indexing="ij")

    head_b_fp64 = copy.deepcopy(head_b).double()
    with torch.no_grad():
        head_b_fp64.a_proj.weight.zero_()
        head_b_fp64.a_proj.bias.zero_()
        head_b_fp64.p_proj.weight.zero_()
        head_b_fp64.p_proj.bias.zero_()
        head_b_fp64.mean_head.weight.zero_()
        head_b_fp64.mean_head.bias.zero_()
        head_b_fp64.phi_proj.weight.zero_()
        head_b_fp64.phi_proj.weight[0, 0, 1, 1, 1] = 1.0
        head_b_fp64.phi_proj.bias.zero_()

    # Linear harmonic phi = 0.5*x - 0.25*y + 0.1*z => Laplacian == 0
    phi_harm = 0.5 * xg - 0.25 * yg + 0.1 * zg
    feat_harm = torch.stack([phi_harm, torch.zeros_like(phi_harm), torch.zeros_like(phi_harm)], dim=0).unsqueeze(0)
    out_harm = head_b_fp64(feat_harm, fluid_mask=mask_t.double(), spacing_zyx=spacing)
    assert float(out_harm["harmonic_laplacian_loss"].item()) < 1e-20

    # Non-harmonic phi = x^2 + y^2 + z^2 => Laplacian == 6 => loss > 0
    phi_nonharm = xg.pow(2) + yg.pow(2) + zg.pow(2)
    feat_nonharm = torch.stack(
        [phi_nonharm, torch.zeros_like(phi_nonharm), torch.zeros_like(phi_nonharm)], dim=0
    ).unsqueeze(0)
    out_nonharm = head_b_fp64(feat_nonharm, fluid_mask=mask_t.double(), spacing_zyx=spacing)
    assert out_nonharm["harmonic_laplacian_loss"].item() == pytest.approx(36.0)

    # 2. End-to-end gradient flow into phi_proj
    head_train = VectorPotentialCurlHead3D(in_channels=8, separate_mean_potential=True, use_potential_branch=True)
    feat_req = torch.randn(2, 2, 8, *shape_zyx, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = head_train(feat_req, fluid_mask=mask_t.expand(2, *shape_zyx), spacing_zyx=spacing)
    assert out["potential_phi"] is not None and out["potential_phi"].shape == (2, 2, *shape_zyx)
    loss = out["harmonic_laplacian_loss"]
    assert loss.dtype == torch.float32
    loss.backward()
    assert feat_req.grad is not None and torch.isfinite(feat_req.grad).all()
    assert head_train.phi_proj is not None and head_train.phi_proj.weight.grad is not None
    assert float(head_train.phi_proj.weight.grad.abs().sum().item()) > 0.0


@pytest.mark.parametrize("immersed_solid", [False, True])
def test_head_b_harmonic_loss_matches_divergence_of_output_velocity(immersed_solid):
    coord = torch.arange(-4, 5, dtype=torch.float64)
    _, y, x = torch.meshgrid(coord, coord, coord, indexing="ij")
    # At unit spacing the seven-point Laplacian is identically zero, but
    # applying the centered divergence to the centered gradient gives 12.
    phi = x**4 - 6 * x.square() * y.square() + y**4 - 2 * x.square()
    mask = None
    if immersed_solid:
        mask = torch.ones_like(phi)
        mask[:, 4, 4] = 0
        phi[:, 4, 4] = 1e6  # inaccessible in a complete two-derivative fluid stencil
    head = VectorPotentialCurlHead3D(1, separate_mean_potential=False, use_potential_branch=True).double()
    with torch.no_grad():
        head.a_proj.weight.zero_()
        head.a_proj.bias.zero_()
        head.phi_proj.weight.zero_()
        head.phi_proj.weight[0, 0, 1, 1, 1] = 1
        head.phi_proj.bias.zero_()

    out = head(phi[None, None], fluid_mask=mask, spacing_zyx=(1, 1, 1))
    valid = torch_valid_div_of_curl_mask(torch.ones_like(phi) if mask is None else mask).bool()
    measured = out["div_residual"][0][valid]
    torch.testing.assert_close(measured, torch.full_like(measured, 12))
    torch.testing.assert_close(out["harmonic_laplacian_loss"], measured.square().mean())


def test_head_losses_fail_when_a_frame_has_no_complete_fluid_stencil():
    feat = torch.zeros(2, 1, 9, 9, 9)
    mask = torch.ones(2, 9, 9, 9)
    # One layer permits no first derivatives. Three layers permit first but
    # not two successive centered derivatives, even if another frame is valid.
    mask[1, 1:] = 0
    with pytest.raises(PhysicsContractError, match="complete interior fluid stencil"):
        DirectFieldHead3D(1)(feat, fluid_mask=mask)
    mask[1, :3] = 1
    with pytest.raises(PhysicsContractError, match="complete interior fluid stencil"):
        VectorPotentialCurlHead3D(1, use_potential_branch=True)(feat, fluid_mask=mask)


@pytest.mark.parametrize("projection", [False, True])
def test_head_a_rejects_invalid_predictor_before_divergence(projection):
    head = DirectFieldHead3D(1, helmholtz_projection=projection)
    with torch.no_grad():
        head.proj.bias[:3].fill_(float("nan"))
    with pytest.raises(PhysicsContractError, match="non-finite direct field"):
        head(torch.ones(1, 1, 8, 8, 8))


@pytest.mark.parametrize("branch", ["a_proj", "phi_proj", "p_proj"])
def test_head_b_rejects_nonfinite_network_outputs(branch):
    head = VectorPotentialCurlHead3D(1, use_potential_branch=True)
    with torch.no_grad():
        getattr(head, branch).bias.fill_(float("inf"))
    with pytest.raises(PhysicsContractError, match="non-finite"):
        head(torch.ones(1, 1, 8, 8, 8))


def test_projection_scrubs_only_known_solid_nan_and_rejects_fluid_nan():
    vel = torch.ones(3, 8, 8, 8)
    mask = torch.ones(8, 8, 8)
    mask[3, 3, 3] = 0
    vel[:, 3, 3, 3] = float("nan")
    projected, _ = torch_helmholtz_project(vel, (1, 1, 1), mask, num_iters=2)
    assert torch.isfinite(projected).all()
    assert not projected[:, 3, 3, 3].any()
    assert torch.isfinite(torch_div_h(*vel, (1, 1, 1), mask)).all()
    vel[:, 5, 5, 5] = float("nan")
    assert not torch.isfinite(torch_div_h(*vel, (1, 1, 1), mask)).all()
    with pytest.raises(PhysicsContractError, match="non-finite projection velocity"):
        torch_helmholtz_project(vel, (1, 1, 1), mask, num_iters=2)


def test_first_derivative_head_losses_average_only_valid_cells_per_frame():
    x = torch.arange(9, dtype=torch.float32).expand(9, 9, 9)
    feat = torch.stack([x, 3 * x])[:, None]
    mask = torch.ones_like(feat[:, 0])
    mask[1, 4:] = 0
    direct = DirectFieldHead3D(1)
    curl = VectorPotentialCurlHead3D(1, separate_mean_potential=False)
    with torch.no_grad():
        for proj in (direct.proj, curl.a_proj):
            proj.weight.zero_()
            proj.weight[0, 0, 1, 1, 1] = 1
            proj.bias.zero_()
    # Per-frame squared divergences 1 and 9 contribute equally despite the
    # different number of valid cells, also matching uneven multi-GPU shards.
    assert direct(feat, mask, (1, 1, 1))["div_loss"].item() == pytest.approx(5.0)
    assert curl(feat, mask, (1, 1, 1))["coulomb_gauge_loss"].item() == pytest.approx(5.0)


def test_broadcast_mask_to_spatial_4d_1zyx_to_5d_btzyx():
    shape_zyx = (8, 12, 16)
    spacing = (0.0625, 0.0625, 0.0625)
    geom_np, _ = make_cylinder_geometry_3d(shape_zyx, spacing, cylinder_diameter_m=0.3)
    mask_1zyx = torch.from_numpy(geom_np[0]).unsqueeze(0)  # (1, Z, Y, X)

    bcast = _broadcast_mask_to_spatial(mask_1zyx, (3, 2, *shape_zyx), mask_1zyx.device)
    assert bcast.shape == (1, 1, *shape_zyx)

    u = torch.randn(3, 2, *shape_zyx)
    v = torch.randn(3, 2, *shape_zyx)
    w = torch.randn(3, 2, *shape_zyx)
    div_bcast = torch_div_h(u, v, w, spacing, mask=mask_1zyx)
    div_full = torch_div_h(u, v, w, spacing, mask=mask_1zyx.unsqueeze(1).expand(3, 2, *shape_zyx))
    assert torch.allclose(div_bcast, div_full, atol=1e-6, rtol=1e-6)

    with pytest.raises(PhysicsContractError, match="cannot broadcast mask shape"):
        _broadcast_mask_to_spatial(mask_1zyx.expand(4, *shape_zyx), (3, 2, *shape_zyx), mask_1zyx.device)


def test_forward_elastic_multi_gpu_boundary_cases_and_autocast_parity():
    plan = resolve_elastic_runtime_plan({"runtime": {"num_gpus": "auto"}})
    spacing = (0.0625, 0.0625, 0.0625)

    # Validation checks on invalid device_ids & input batch mismatches
    _, x_small, geom_small, b_small = _make_small_batch(bsz=2, t_len=1, shape_zyx=(16, 24, 32), spacing=spacing)
    model_val = Flow3DStage1Autoencoder(
        encoder=DCAE3DEncoder(),
        spatial_transformer=SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024),
        latent_norm=LatentNorm3D(channels=256, mode="group_norm"),
        decoder=DCAE3DDecoder(),
        head=DirectFieldHead3D(in_channels=32, helmholtz_projection=False),
        readout=PhysicalReadout3D(),
        default_spacing_zyx=spacing,
    )
    with pytest.raises(ValueError, match="non-empty"):
        model_val.forward_elastic_multi_gpu(x_small, geom_small, b=b_small, device_ids=[])
    with pytest.raises(ValueError, match="unique"):
        model_val.forward_elastic_multi_gpu(x_small, geom_small, b=b_small, device_ids=[0, 0])
    with pytest.raises(ValueError, match="non-negative"):
        model_val.forward_elastic_multi_gpu(x_small, geom_small, b=b_small, device_ids=[-1])
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="out-of-range"):
            model_val.forward_elastic_multi_gpu(x_small, geom_small, b=b_small, device_ids=[999])
    else:
        with pytest.raises(ValueError, match="CUDA is not available"):
            model_val.forward_elastic_multi_gpu(x_small, geom_small, b=b_small, device_ids=[999])
    with pytest.raises(ValueError, match="batch size B must be >= 1"):
        model_val.forward_elastic_multi_gpu(x_small[:0], geom_small[:0], b=b_small[:0], num_gpus=0)
    with pytest.raises(ValueError, match="geometry batch size"):
        model_val.forward_elastic_multi_gpu(x_small, geom_small[:1], b=b_small, num_gpus=0)
    with pytest.raises(ValueError, match="b statistics batch size"):
        model_val.forward_elastic_multi_gpu(x_small, geom_small, b=b_small[:1], num_gpus=0)

    primary_dev = torch.device(plan.primary_device)

    # 1. n=1 single GPU & B=1 < n boundary
    _, x1, g1, b1 = _make_small_batch(bsz=1, t_len=1, shape_zyx=(16, 24, 32), spacing=spacing, seed=11)
    model_val.eval()
    out_n1 = model_val.forward_elastic_multi_gpu(x1, g1, b=b1, num_gpus=1)
    assert out_n1["elastic_runtime"]["active_gpus"] == (1 if plan.active_gpus >= 1 else 0)
    assert out_n1["phys_fields"].shape == (1, 1, 4, 16, 24, 32)

    out_b1_lt_n = model_val.forward_elastic_multi_gpu(x1, g1, b=b1, num_gpus="auto")
    assert out_b1_lt_n["elastic_runtime"]["active_gpus"] == (1 if plan.active_gpus >= 1 else 0)
    assert out_b1_lt_n["elastic_runtime"]["shard_batch_sizes"] == [1]

    # 2. B=3 < n (uses 3 GPUs on 5-GPU host) with Head-A & T=2
    _, x3_t2, g3_t2, b3_t2 = _make_small_batch(bsz=3, t_len=2, shape_zyx=(16, 24, 32), spacing=spacing, seed=13)
    out_b3 = model_val.forward_elastic_multi_gpu(x3_t2, g3_t2, b=b3_t2, num_gpus="auto")
    expected_b3_gpus = min(plan.active_gpus, 3) if plan.active_gpus >= 1 else 0
    assert out_b3["elastic_runtime"]["active_gpus"] == expected_b3_gpus
    assert sum(out_b3["elastic_runtime"]["shard_batch_sizes"]) == 3
    assert out_b3["phys_fields"].shape == (3, 2, 4, 16, 24, 32)

    # 3. bfloat16 autocast preservation inside parallel_apply worker threads (Head-A norm_fields stays bfloat16)
    if plan.active_gpus >= 2:
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ref_bf16 = model_val.to(primary_dev)(x3_t2.to(primary_dev), g3_t2.to(primary_dev), b=b3_t2.to(primary_dev))
            el_bf16 = model_val.forward_elastic_multi_gpu(x3_t2, g3_t2, b=b3_t2, num_gpus="auto")
        assert el_bf16["norm_fields"].dtype == torch.bfloat16
        assert torch.allclose(ref_bf16["norm_fields"], el_bf16["norm_fields"], atol=5e-2, rtol=5e-2)

    # 4. Uneven B=7 across n=active_gpus (e.g. [2, 2, 1, 1, 1] on 5 GPUs): verify loss & grad parity vs single-GPU
    _, x7, g7, b7 = _make_small_batch(bsz=7, t_len=1, shape_zyx=(16, 24, 32), spacing=spacing, seed=17)
    torch.manual_seed(123)
    model_b = Flow3DStage1Autoencoder(
        encoder=DCAE3DEncoder(),
        spatial_transformer=SpatialTransformer3D(dim=256, depth=1, num_heads=8, mlp_dim=1024),
        latent_norm=LatentNorm3D(channels=256, mode="group_norm"),
        decoder=DCAE3DDecoder(),
        head=VectorPotentialCurlHead3D(in_channels=32, use_potential_branch=True),
        readout=PhysicalReadout3D(),
        default_spacing_zyx=spacing,
    ).to(primary_dev)
    model_b.train()

    # Single-GPU reference backward
    model_b.zero_grad(set_to_none=True)
    ref_7 = model_b(x7.to(primary_dev), g7.to(primary_dev), b=b7.to(primary_dev))
    ref_loss = (
        ref_7["phys_fields"].pow(2).mean()
        + 0.01 * ref_7["head_aux"]["coulomb_gauge_loss"]
        + 0.01 * ref_7["head_aux"]["harmonic_laplacian_loss"]
    )
    ref_loss.backward()
    ref_grads = [p.grad.detach().clone() for p in model_b.parameters() if p.requires_grad]

    # Elastic n-GPU uneven-shard backward
    model_b.zero_grad(set_to_none=True)
    el_7 = model_b.forward_elastic_multi_gpu(x7, g7, b=b7, num_gpus="auto")
    el_loss = (
        el_7["phys_fields"].pow(2).mean()
        + 0.01 * el_7["head_aux"]["coulomb_gauge_loss"]
        + 0.01 * el_7["head_aux"]["harmonic_laplacian_loss"]
    )
    el_loss.backward()
    el_grads = [p.grad.detach().clone() for p in model_b.parameters() if p.requires_grad]

    expected_m7 = min(plan.active_gpus, 7) if plan.active_gpus >= 1 else 0
    assert el_7["elastic_runtime"]["active_gpus"] == expected_m7
    assert sum(el_7["elastic_runtime"]["shard_batch_sizes"]) == 7
    assert torch.allclose(ref_loss, el_loss, atol=1e-4, rtol=1e-4)
    for rg, eg in zip(ref_grads, el_grads):
        assert torch.allclose(rg, eg, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("head_type", ["head_a", "head_b"])
@pytest.mark.parametrize("backend", ["cpu", "two_gpus"])
def test_trainer_gather_fields_and_gradients(head_type, backend):
    from flow3d.models.heads import _trainer_shard_output

    multi_gpu = backend == "two_gpus"
    if multi_gpu and torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    device = torch.device("cuda:0" if multi_gpu else "cpu")
    cfg = {
        "encoder": {"base_channels": 8, "latent_channels": 32},
        "spatial_transformer": {"num_heads": 4, "mlp_dim": 64},
        "latent_norm": {"num_groups": 4},
        "head": {"type": head_type, "use_potential_branch": True},
    }
    torch.manual_seed(27)
    model = Flow3DStage1Autoencoder.from_config(cfg).to(device)
    _, x, geometry, b = _make_small_batch(bsz=3, t_len=1, shape_zyx=(8, 16, 24))
    x, geometry, b = x[:, 0].to(device), geometry.to(device), b[:, 0].to(device)

    def forward(phase=None):
        if multi_gpu:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return model.forward_elastic_multi_gpu(x, geometry, b=b, device_ids=[0, 1], gather_for=phase)
        full = model(x, geometry, b=b)
        return full if phase is None else _trainer_shard_output(full, phase)

    def gradients(output):
        # Include all auxiliary terms consumed by train_step, including the optional Head B branch.
        pred = output["norm_fields"].float()
        loss = (pred - x).square().mean()
        for name in ("div_loss", "coulomb_gauge_loss", "harmonic_laplacian_loss"):
            term = output["head_aux"].get(name)
            if term is not None:
                loss = loss + 0.01 * term
        model.zero_grad(set_to_none=True)
        loss.backward()
        return {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}

    full = forward()
    selected = forward("train")
    expected_keys = {"norm_fields", "head_aux"} | ({"elastic_runtime"} if multi_gpu else set())
    assert set(selected) == expected_keys
    if multi_gpu:
        assert selected["elastic_runtime"]["shard_batch_sizes"] == [2, 1]
    loss_names = {"div_loss", "coulomb_gauge_loss", "harmonic_laplacian_loss"} & full["head_aux"].keys()
    assert set(selected["head_aux"]) == loss_names
    for key in loss_names:
        torch.testing.assert_close(selected["head_aux"][key], full["head_aux"][key])
    torch.testing.assert_close(selected["norm_fields"], full["norm_fields"])
    reference_grads, selected_grads = gradients(full), gradients(selected)
    assert reference_grads.keys() == selected_grads.keys()
    for name in reference_grads:
        torch.testing.assert_close(reference_grads[name], selected_grads[name], atol=1e-5, rtol=1e-4)

    model.eval()
    with torch.no_grad():
        full = forward()
        selected = forward("validation")
    assert set(selected) == {"norm_fields", "phys_fields"} | ({"elastic_runtime"} if multi_gpu else set())
    for key in ("norm_fields", "phys_fields"):
        torch.testing.assert_close(selected[key], full[key])
