# 给 Codex / Coding AI 的首轮任务：FlowWorld-3D V1.3（路线 A）· R1 / G0

你在本地主仓库 `D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-WorldModel`（Git 远端 `Pitohuie-AIversion/HydroDisturb-WorldModel.git`）中执行任务。完整工程设计见同目录 `docs/FlowWorld3D_V1_Development_Design.md`。
**本轮仅实现 R1 / G0：三维流场数据契约、本机真实 CFD 数据探针与适配器、防泄漏分组切分、逐帧有效流体域六维归一化、相容离散 `div_h / curl_h` 算子与真实壁面边界检验接口。** 严禁提前实现或训练 Stage 1 神经网络、CLIP 或 Stage 2 动力学模型。

---

## 1. 当前仓库与本机数据已知事实

1. **仓库现状（路线 A）**：
   - 当前仓库已有 `src/stage1/`、`configs/stage1.yaml` 及 `tests/test_*.py`（6 个测试文件）。其中 `src/stage1/io/checkpoint.py` 是已验证的成熟原子检查点与 SHA-256 冻结契约模块。
   - 本轮在 **`src/flow3d/`、`configs/flow3d/`、`tests/flow3d/`** 下构建正式 V1.3 的 G0 基础模块，**不得破坏既有 `src/stage1/` 与 `tests/test_*.py`**。
   - 当前 Python 环境基础依赖为 `numpy` 与 `pytest`（`PyTorch`、`pyyaml`、`h5py` 可能未预装，代码必须保证在仅安装 `numpy` 时核心 G0 契约、探针、归一化、物理算子与全部 R1 单元测试均可 100% 运行通过，绝不允许将新增测试标为 `skip`）。
2. **本机实测存在的 3 套三维 CFD 数据源**（探针与适配器必须全部支持并区分）：
   - **数据源 A（HydroDisturb-Bench C04 基准 4 帧体场）**：
     `D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-Bench\runs\c04_s1_volume_sequence.npz`
     - 含 `fields (4, 4, 64, 96, 192) float32`（`[u, v, w, p]`，单位 $\text{m/s}, \text{Pa}$）、`solid_mask (64, 96, 192) bool`（**`True=solid`，须翻转为 `fluid_mask = 1 - solid_mask`**）、`sdf (64, 96, 192) float32`（**米制**，须除以 `cylinder_diameter_m = 0.3` 得无量纲 `SDF/D`）、`times (4,)`（$\Delta t = 0.06\,\text{s}$）、`metadata_json`（`cell_spacing_m = 0.01875`）。
   - **数据源 B（C04 涡脱落 82 帧原始体场目录）**：
     `Z:\E\cfd_3d\05_reports_viz\video_c04_shedding\c04_vol_step_*.npz`（82 个文件）
     - 每个文件含 `u, v_span, w_cross, p`，形状为 **`(96, 64, 192)` = `(Y_cross=96, Z_span=64, X_stream=192)`**（前两轴相对 `(Z=64, Y=96, X=192)` 转置！），且圆柱固体内部含 **12,800 个 `NaN`（`finite=False`）**；
     - 转换为标准 `[u, v, w, p]` `(4, Z=64, Y=96, X=192)` 时，必须对空间维执行 `transpose(1, 0, 2)`，且将速度分量映射为 **`u_std = u.T(1,0,2)`（沿 $X$）、`v_std = w_cross.T(1,0,2)`（沿 $Y$）、`w_std = v_span.T(1,0,2)`（沿 $Z$）**；
     - 82 帧的 `time_s` 存在重复时间戳（`dt=0.0s` 1 处）及多档非均匀步长（`0.12s, 0.24s, 0.96s, 1.02s, 1.98s, 2.04s`），必须识别并拆分为连续等步长子段（`segment_id`）。
   - **数据源 C（多雷诺数世界模型 HDF5 数据集与清单）**：
     `Z:\E\cfd_3d\03_world_model_dataset\manifest.json` 及 `multi_re_tier1_certification_summary.json`、`extraction_and_rampool_assessment.json`
     - 清单记录 4 条 HDF5 轨迹（`(T, 6, 64, 96, 192)`），其 6 通道顺序为 `["u_star", "v_cross_star", "w_span_star", "p_star", "sdf_star", "solid_mask"]`（**第 4 通道为 `sdf_star`，第 5 通道为 `solid_mask` `1=solid`**，需重排并翻转为 `[fluid_mask, SDF/D]`）；
     - **必须自动识别并隔离伪数据/错配数据**：`c04_wm_25steps_golden.h5` 的 `ground_truth_verified == false` 且含 `ground_truth_warning`（实为 `Re=5000` 启动前 2 帧的秩-2 线性插值伪数据被误标为 `Re=126167`），`re180k_mesh1_pilot` 实际求解 $U_0=0.5\,\text{m/s}$ 与目标 $Re=180,000$ 冲突。探针必须将其标记为 `QUARANTINED` 并禁止进入训练集！

---

## 2. 必须先做的调查

1. 运行 `git status --short`、`git rev-parse HEAD`，确认当前工作树状态；不得覆盖或删除已有未提交目录（如 `flowscope-3d-v2/`、`site-publish/`、`docs/`）。
2. 阅读 `docs/FlowWorld3D_V1_Development_Design.md`、`src/stage1/config.py`、`src/stage1/data/normalization.py`、`src/stage1/physics/operators.py`、`src/stage1/io/checkpoint.py` 及 `tests/test_*.py`。
3. 检查本机上述 3 套数据源路径的可访问性与元数据，记录实测结果。

---

## 3. 本轮具体实现范围（R1 / G0）

创建并实现以下模块（禁止提交空文件或占位 `pass`）：

1. **`src/flow3d/__init__.py` & `src/flow3d/contracts.py`**：
   - 定义 `StateSpec3D` 与 `CYLINDER3D_STATE_SPEC`（物理通道 `("u", "v", "w", "p")`，几何通道 `("fluid_mask", "sdf_over_d")`，`spatial_dim=3`，标准网格 `(64, 96, 192)`，轴序 `(Z, Y, X)` 对应物理坐标 `(z_span, y_cross, x_stream)`）。
   - 定义强类型数据结构与校验器：
     - `SampleProvenance`：包含完整字段 `case_id, geometry_id, trajectory_id, ancestor_id, segment_id, time_idx, time_phys_s, time_ctu, ground_truth_verified, tier1_stability_status`；
     - `Flow3DConditions`：包含 `reynolds_number, u0_m_s, diameter_m, viscosity_m2_s, dt_phys_s, dt_ctu, action_schedule, text_prompt, known_mask, valid_mask`；当 `reynolds_number` 与 `(u0_m_s, diameter_m, viscosity_m2_s)` 同时给定时，**强制校验 $Re \approx |U_0| D / \nu$（相对误差超过阈值时拒绝或显式抛出 `PhysicsContractError`）**；严禁未来不可见字段进入推理条件；
     - `Flow3DBatch`：校验 `field [B, T, 4, Z, Y, X]`、`geometry [B, 2, Z, Y, X]`、`b [B, T, 6]`、`spacing (dz, dy, dx)`。
2. **`src/flow3d/data/schema.py`**：
   - 定义轴序转换契约（标准 `ZYX (64,96,192)` vs 原始 `YZX (96,64,192)`）、速度分量物理对齐映射、单位制（有量纲 SI vs 无量纲 `*_star`）、SDF 符号与无量纲化约定（`SDF/D`，流体 $>0$、壁面 $=0$、固体 $<0$）、边界条件枚举及 JSON/dict 可序列化 schema（提供零第三方依赖的 YAML/JSON 配置读取支持，使 `configs/flow3d/data.yaml` 在无 `pyyaml` 时也能解析）。
3. **`src/flow3d/data/normalization.py`**：
   - 实现 `Flow3DNormalizer` 与 `Flow3DNormStats`（版本标识 `"normalizer.flow3d.v1"`）：
     - 在每帧有效流体域 $M = (\text{fluid\_mask} > 0.5)$ 上以 `float64` 计算 **population variance (`ddof=0`)**，输出 `float32` 的 `b = [mu_u, mu_v, mu_w, mu_p, sigma_uvw, sigma_p]`；
     - **联合速度方差不除以 3**：$\sigma_{uvw} = \max(\sqrt{\operatorname{Var}_M(u) + \operatorname{Var}_M(v) + \operatorname{Var}_M(w)}, \text{scale\_floor}_{uvw})$；压力独立标准差 $\sigma_p = \max(\sqrt{\operatorname{Var}_M(p)}, \text{scale\_floor}_p)$；
     - **前置 `isfinite` 硬校验**：必须在掩码索引前校验有效流体域 $M$ 内所有值有限（流体域含 `NaN/Inf` 直接抛 `ValueError`）；固体域 $\neg M$ 内的 `NaN/Inf` 在索引外清洗为 `0.0`，**严禁使用 `values * mask` 导致 `NaN * 0` 污染**；
     - 空流体掩码（$|M| = 0$）直接 fail closed；常数场零方差在 `fit()` 中直接返回版本化 `scale_floor`（绝不返回 `0.0`）；支持 `[4, Z, Y, X]`、`[T, 4, Z, Y, X]`、`[B, T, 4, Z, Y, X]` 逐帧 `normalize / denormalize / roundtrip`。
4. **`src/flow3d/data/adapters.py` & `src/flow3d/data/inspect.py`**：
   - `adapters.py`：显式支持并转换本机 3 类格式（未知格式明确报错，绝不盲猜）：
     1. `load_c04_benchmark_npz`（读取 `c04_s1_volume_sequence.npz`，执行 `fluid_mask = 1 - solid_mask`、`sdf_over_d = sdf / D`）；
     2. `load_c04_shedding_directory`（读取 `video_c04_shedding/*.npz`，按 `time_s` 排序，剔除或报告重复时间戳，切分恒定 $\Delta t$ 连续子段 `segment_id`，执行 `(96,64,192) -> (64,96,192)` 转置与 `[u, w_cross, v_span, p]` 通道对齐，由固体 `NaN` 提取 `fluid_mask` 并清洗固体 `NaN`）；
     3. `inspect_world_model_dataset_manifest`（解析 `03_world_model_dataset/manifest.json` 与 `multi_re_tier1_certification_summary.json`，映射 `["u_star", "v_cross_star", "w_span_star", "p_star", "sdf_star", "solid_mask"]` 第 4/5 通道互换与极性翻转；当环境装有 `h5py` 时支持直接读取 `.h5` 张量，未装 `h5py` 时解析 manifest + HDF5 头部签名并明确报告）；
     4. **伪数据/错配自动隔离器**：对 `ground_truth_verified == false`、`ground_truth_warning` 非空、$Re \neq |U_0|D/\nu$、或时间序列差分矩阵秩 $\le 2$（线性插值合成伪轨迹，如 `c04_wm_25steps_golden.h5`）自动标记 `quarantined = True` 并说明原因。
   - `inspect.py`：对给定路径生成结构化 G0 诊断报告（形状、轴序、步长 $\Delta t_{\text{phys}}$ 与 $\Delta t_{\text{ctu}}$、有效流体比例、固体 `NaN` 数量、物理散度基线、近壁第一层流速统计、窗口可用性、隔离列表及对 $H=10/50/100$ 的实际支撑上限）。
5. **`src/flow3d/data/splits.py`**：
   - 按 `ancestor_id` 优先、`trajectory_id` 次之进行分组切分，严禁按滑动窗口随机打乱造成 train/val/test 泄漏；
   - 针对单轨迹长序列（或少轨迹场景），额外提供**带安全隔离带（`purge_gap_frames >= L + H`）的连续时间段切分模式（`temporal_block_split_with_purge`）**，并在 provenance 缺失时拒绝宣称零泄漏。
6. **`src/flow3d/physics/operators.py` & `src/flow3d/physics/boundary.py`**：
   - `operators.py`：在 `(..., Z, Y, X)` + `spacing = (dz, dy, dx)` 约定下实现 `centered_difference`、`div_h`（$\partial_X u + \partial_Y v + \partial_Z w$）、三维旋度 `curl_h`（$(\partial_Y A_z - \partial_Z A_y,\ \partial_Z A_x - \partial_X A_z,\ \partial_X A_y - \partial_Y A_x)$）及有效导数掩码 `build_3d_derivative_mask`（含算子级联后的二次腐蚀掩码 `valid_div_of_curl_mask`）；
   - 提供解析场检验函数，验证在相容中心差分与二次有效内部掩码 `valid_div_of_curl_mask` 上离散恒等式 $\|\operatorname{div}_h(\operatorname{curl}_h(\mathbf{A}))\|_\infty \le \epsilon$ 成立，并显式证明/记录在圆柱固体边界 1–2 层截断处该恒等式不自动成立；
   - `boundary.py`：提取圆柱固体表面相邻的**第一层真实流体体素掩码（`first_fluid_shell_mask`）**，结合 `SDF/D` 梯度估算局部壁面法向 $\mathbf{n}$ 与切向速度分量 $(u_n, u_\tau)$，以及入口（$X=0$）/出口（$X=-1$）边界通量与速度剖面检查接口（明确区分“固体内部填零”与“真实壁面第一层流体无滑移/无穿透接近度”）。
7. **`src/flow3d/data/fixtures.py`、`configs/flow3d/data.yaml`、`src/flow3d/cli/inspect_data.py`**：
   - `fixtures.py`：提供 3D 解析无散场（如由光滑矢量势 $\mathbf{A}$ 经 `curl_h` 生成的精确离散无散场 + 3D Taylor-Green 涡解析场）与带圆柱障碍物/固体 `NaN`/线性插值伪轨迹的合成测试 fixture；
   - `configs/flow3d/data.yaml`：记录 C04 标准三维网格、物理常数与本机 3 套数据源路径配置；
   - `cli/inspect_data.py`：支持 `--help`、`--path`、`--config`、`--json-out`，可对本机真实 C04 NPZ、82 帧脱落目录与 `03_world_model_dataset` 输出完整 G0 诊断报告。
8. **`tests/flow3d/` 四套完整单元测试**：
   - `tests/flow3d/test_contracts.py`
   - `tests/flow3d/test_normalization.py`
   - `tests/flow3d/test_axes_and_geometry.py`
   - `tests/flow3d/test_physics_operators.py`

---

## 4. 必须执行的验证命令

```powershell
python -m pytest tests/flow3d/test_contracts.py tests/flow3d/test_normalization.py -q
python -m pytest tests/flow3d/test_axes_and_geometry.py tests/flow3d/test_physics_operators.py -q
python -m pytest tests/flow3d -q
python -m pytest tests -q
python -m src.flow3d.cli.inspect_data --help
python -m src.flow3d.cli.inspect_data --config configs/flow3d/data.yaml
```

---

## 5. 验收标准与交付证据

1. 全部新增测试（`tests/flow3d/`）与既有测试（`tests/test_*.py`）真实运行且 100% 通过（0 failed, `tests/flow3d/` 0 skipped）。
2. `python -m src.flow3d.cli.inspect_data --config configs/flow3d/data.yaml` 在本机真实 3 套数据源上成功运行，如实报告：
   - `c04_s1_volume_sequence.npz`（4 帧，`solid_mask` 翻转、`SDF/D` 换算、真实物理散度 `RMS ≈ 0.1078 s⁻¹`）；
   - `video_c04_shedding`（82 帧，`(96,64,192)` 转置、`w_cross/v_span` 通道对齐、12,800 个固体 `NaN` 安全清洗、非均匀 $\Delta t$ 子段切分）；
   - `03_world_model_dataset`（3 条 50 步有效多雷诺数轨迹 `re200k/re350k/re1m` 通过，伪数据 `c04_wm_25steps_golden.h5` 与错配 `re180k` 被自动标记为 `QUARANTINED`）。
3. 回复需包含：`git status`、新增/修改文件清单、关键接口签名、每条测试命令的真实输出、本机 3 套真实数据的 G0 探针实测摘要、未解决问题（P0–P3）及下一轮（R2）唯一最小任务。
