from pathlib import Path

import pytest

from flow3d.models import heads


@pytest.fixture
def proc(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 268435456 kB\nMemAvailable: 134217728 kB\n")
    monkeypatch.setattr(heads, "_PROC", proc)
    monkeypatch.setattr(heads.os, "cpu_count", lambda: 208)
    monkeypatch.setattr(heads.os, "sched_getaffinity", lambda _: set(range(32)), raising=False)
    return proc


def test_unified_ancestor_cpu_quota_and_remaining_memory(tmp_path, proc):
    mount = tmp_path / "cgroup"
    leaf = mount / "trainer"
    leaf.mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/tenant/trainer\n")
    (proc / "self/mountinfo").write_text(f"12 1 0:12 /tenant {mount} ro - cgroup2 cgroup rw\n")
    (leaf / "cpu.max").write_text("max 100000")
    (mount / "cpu.max").write_text("2200000 100000")
    for directory, limit, usage in [(mount, 110, 20), (leaf, 100, 5)]:
        (directory / "memory.max").write_text(str(limit * 1024 ** 3))
        (directory / "memory.current").write_text(str(usage * 1024 ** 3))

    plan = heads.resolve_elastic_runtime_plan(detected_gpus=0)
    assert plan.detected_cpus == plan.cpu_threads_total == 22
    assert plan.detected_ram_gib == 90


def test_legacy_cpu_and_memory_controllers(tmp_path, proc):
    cpu, memory = tmp_path / "cpu", tmp_path / "memory"
    cpu.mkdir()
    memory.mkdir()
    (proc / "self/cgroup").write_text("3:cpu,cpuacct:/\n4:memory:/\n")
    (proc / "self/mountinfo").write_text(
        f"12 1 0:12 / {cpu} ro - cgroup cgroup rw,cpu,cpuacct\n"
        f"13 1 0:13 / {memory} ro - cgroup cgroup rw,memory\n"
    )
    (cpu / "cpu.cfs_quota_us").write_text("150000")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    (memory / "memory.limit_in_bytes").write_text(str(8 * 1024 ** 3))
    (memory / "memory.usage_in_bytes").write_text(str(3 * 1024 ** 3))
    plan = heads.resolve_elastic_runtime_plan(detected_gpus=0)
    assert plan.cpu_threads_total == 2
    assert plan.detected_ram_gib == 5


def test_cpu_affinity_and_host_memory_without_cgroups(proc: Path):
    plan = heads.resolve_elastic_runtime_plan(detected_gpus=0)
    assert plan.cpu_threads_total == 32
    assert plan.detected_ram_gib == 128
