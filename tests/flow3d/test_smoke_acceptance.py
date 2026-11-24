import runpy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flow3d.models import LatentPhysicsProbe, VectorPotentialCurlHead3D


@pytest.fixture
def smoke():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/smoke_flow3d_stage1.py"))


@pytest.mark.parametrize(
    "args,reason",
    [
        (["--device", "cuda"], "requires at least one active CUDA GPU"),
        (["--device", "auto", "--amp"], "requires CUDA"),
    ],
)
def test_explicit_cuda_or_amp_cannot_pass_cpu_smoke(smoke, monkeypatch, args, reason):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "argv", ["smoke", *args])
    with pytest.raises(ValueError, match=reason):
        smoke["main"]()


def test_enabled_probe_and_gradient_acceptance(smoke):
    probe = LatentPhysicsProbe(latent_channels=8)
    out = {
        "phys_fields": torch.ones(1, requires_grad=True),
        "head_aux": {},
        "probe": probe(torch.randn(1, 8, 2, 2, 2)),
    }
    loss = smoke["_smoke_loss"](out)
    loss.backward()
    checks = smoke["_check_smoke_backward"](probe, loss)
    assert checks["checked_gradient_tensors"] == len(list(probe.parameters()))
    assert checks["allowed_zero_gradient_tensors"] == checks["zero_gradient_tensors"] == []
    # One tensor may disconnect while its sibling in the same head still receives gradients.
    probe.enstrophy_head.bias.grad.zero_()
    assert probe.enstrophy_head.weight.grad.abs().max().item() > 0
    with pytest.raises(RuntimeError, match="zero gradient tensors: enstrophy_head.bias"):
        smoke["_check_smoke_backward"](probe, loss)
    probe.enstrophy_head.bias.grad.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="non-finite gradients: enstrophy_head.bias"):
        smoke["_check_smoke_backward"](probe, loss)
    probe.enstrophy_head.bias.grad = None
    with pytest.raises(RuntimeError, match="missing gradients: enstrophy_head.bias"):
        smoke["_check_smoke_backward"](probe, loss)


@pytest.mark.parametrize("with_phi", [False, True])
def test_only_spatially_constant_potential_biases_may_have_zero_gradients(smoke, with_phi):
    model = torch.nn.Module()
    model.head = VectorPotentialCurlHead3D(in_channels=2, use_potential_branch=with_phi)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model.head.a_proj.bias.grad.zero_()
    expected = ["head.a_proj.bias"]
    if with_phi:
        model.head.phi_proj.bias.grad.zero_()
        expected.append("head.phi_proj.bias")
    checks = smoke["_check_smoke_backward"](model, torch.tensor(0.0))
    assert checks["allowed_zero_gradient_tensors"] == expected
    assert checks["zero_gradient_tensors"] == expected
    model.head.a_proj.weight.grad.zero_()
    with pytest.raises(RuntimeError, match="zero gradient tensors: head.a_proj.weight"):
        smoke["_check_smoke_backward"](model, torch.tensor(0.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CPU and CUDA")
def test_gradient_acceptance_across_parameter_devices(smoke):
    model = torch.nn.ModuleList([torch.nn.Linear(2, 1), torch.nn.Linear(2, 1).cuda()])
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    assert smoke["_check_smoke_backward"](model, torch.tensor(0.0))["checked_gradient_tensors"] == 4
