# FlowWorld-3D V1.3（路线 A 修订版）：三维流场时空世界模型 AI Coding 工程设计

> **主仓库绑定（路线 A）**：`D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-WorldModel`（Git 远端：`Pitohuie-AIversion/HydroDisturb-WorldModel.git`，已完成 R1 / G0 基线提交：`f5dc000`，初始骨架提交：`15aa63b`），服务于同级基准仓 `D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-Bench`（C04 三维高雷诺数圆柱绕流基准）。
> **修订说明**：本版本对齐路线 A，消除原 `Design.md` 与 `R1_Prompt.md` 的 8 处内部口径漂移，接入本机实测的 3 套三维 CFD 数据源契约，并修正了原路线中的 6 项数学、物理与自回归动力学隐患。

---

## A. 仓库现状与路线 A 最小变更策略

1. **主仓与存量模块处置**：
   - 本仓库 `src/stage1/` 已包含第一版协议骨架与**已验证的成熟原子检查点/冻结清单模块 `src/stage1/io/checkpoint.py`**（含 `mkstemp + fsync + os.replace`、SHA-256 指针、Python/NumPy/Torch CPU+CUDA RNG 恢复与 `CheckpointContract` 强校验），以及 6 个单元测试文件（`tests/test_*.py`）。
   - `src/stage1/` 中既有的 3 项 P0 数值缺陷（`physics/operators.py` 与 `torch_operators.py` 的 `u/w` 微分轴反转、`data/normalization.py` 固体区 `NaN * 0` 污染与 `fit()` 零尺度返回、`models/dcae.py` 未监督预测掩码门控与 2 通道几何切片崩溃）已先行修复，确保现有 `tests/test_*.py` 100% 绿色通过。
   - 存量 `src/stage1/models/dcae.py` 与 `src/stage1/training.py` 保留为 Stage 1 原型兼容层（不在 R1 中继续扩写）；正式 V1.3 体系在 **`src/flow3d/`、`configs/flow3d/`、`tests/flow3d/`** 独立命名空间下按 R1–R5 分轮演进，并直接复用 `src/stage1/io/checkpoint.py` 的原子持久化底座。
2. **本机实测三维 CFD 数据源现状（G0 必须显式适配与校验）**：
   - **数据源 1（C04 基准 4 帧体场序列）**：`D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-Bench\runs\c04_s1_volume_sequence.npz`
     - 形状与通道：`fields (4, 4, 64, 96, 192) float32`，物理量纲 `[u, v, w, p]`（$\text{m/s}, \text{Pa}$），`times [0.0, 0.06, 0.12, 0.18] s`（$\Delta t = 0.06\,\text{s}$），`cell_spacing_m = 0.01875`，`cylinder_diameter_m = 0.3`；
     - 几何注意：`solid_mask (64, 96, 192) bool` 为 **`True = solid`（必须翻转为 `fluid_mask = 1 - solid_mask`）**；`sdf` 为**米制** `[-0.1367, 2.9704] m`（必须除以 $D = 0.3\,\text{m}$ 转换为无量纲 `SDF/D`）；
     - 窗口限制：仅 4 帧，可用于 G0/G1 单帧重构与算子核验，但**不足以构造 $L=4, H \ge 1$ 的 Stage 2 窗口**。
   - **数据源 2（C04 脱落过程 82 帧原始体场快照）**：`Z:\E\cfd_3d\05_reports_viz\video_c04_shedding\c04_vol_step_*.npz`（82 个文件，$t \in [0.0, 60.0]\,\text{s}$）
     - 轴序与通道映射：原始数组形状为 **`(96, 64, 192)` = `(Y_cross=96, Z_span=64, X_stream=192)`**（前两轴相对标准 `(Z=64, Y=96, X=192)` 转置！），且键名为 `u`（流向 $X$）、`w_cross`（横向 $Y$）、`v_span`（展向 $Z$）、`p`。适配器必须执行 `transpose(1, 0, 2)` 并将通道映射为 `[u, w_cross, v_span, p]`；
     - 固体区 `NaN` 红线：圆柱固体内部包含 **12,800 个 `NaN`（`finite=False`）**，必须在掩码外先行清洗并在有效流体域上先做 `isfinite` 校验；
     - 非均匀时间步红线：82 帧跨度内 $\Delta t$ **非均匀**（含重复时间戳 `0.0s` 1 处、`0.12s` 25 处、`0.24s` 21 处、`0.96s` 6 处、`1.02s` 10 处、`1.98s` 12 处、`2.04s` 6 处），窗口索引器必须按**连续恒定 $\Delta t$ 子段（Segment）**切分，严禁跨变步长跳变点拼接固定步长窗口。
   - **数据源 3（多雷诺数 HDF5 世界模型数据集）**：`Z:\E\cfd_3d\03_world_model_dataset\*.h5` 及 `manifest.json`、`multi_re_tier1_certification_summary.json`
     - 包含 3 条已通过真值核验的 50 步无量纲轨迹：`re200k_wm_50steps.h5`（$Re=2\times 10^5, \Delta t=0.041\,\text{s}, \Delta t^*=0.108\,\text{CTU}$）、`re350k_wm_50steps.h5`（$Re=3.5\times 10^5, \Delta t=0.02\,\text{s}, \Delta t^*=0.092\,\text{CTU}$）、`re1m_wm_50steps.h5`（$Re=10^6, \Delta t=0.01\,\text{s}, \Delta t^*=0.132\,\text{CTU}$），形状均为 `(50, 6, 64, 96, 192)`；
     - 通道顺序与极性红线：HDF5 内置 6 通道顺序为 `["u_star", "v_cross_star", "w_span_star", "p_star", "sdf_star", "solid_mask"]`——**第 4 通道是 `sdf_star`、第 5 通道是 `solid_mask`（`1=solid`）**，适配器必须将第 4/5 通道重排并翻转为 `[fluid_mask = 1 - solid_mask, SDF/D = sdf_star]`；
     - **伪数据自动隔离红线（Quarantine Gate）**：同目录下的 `c04_wm_25steps_golden.h5`（25 步）在 `manifest.json` 中存在 `ground_truth_verified: false` 及 `ground_truth_warning`（实为 `Re=5000` 启动瞬态前 2 帧的秩-2 线性插值伪数据并被误标为 `Re=126167`），以及 `re180k_mesh1_pilot` 存在求解器 $U_0=0.5\,\text{m/s}$ 配置错配。G0 `inspect.py` 与 `adapters.py` 必须**强制拦截并隔离** `ground_truth_verified == false`、`Re ≈ |U0|D/nu` 冲突或时间差分秩 $\le 2$ 的合成插值轨迹，严禁混入训练或评估！

---

## B. 正式总体架构（消除数学与物理隐患后的 V1.3 定型版）

```text
CFD [u,v,w,p] + ZYX Grid (64×96×192) + Geometry [fluid_mask, SDF/D] + Provenance/Conditions
     │
     ▼
G0: inspect/validate axes (ZYX vs YZX), units, Re≈|U0|D/nu, solid NaNs, dt segments,
    quarantine rank-2/mismatched runs, group splits (ancestor/trajectory + temporal purge gap)
     │
     ▼
Stage 1: valid-fluid per-frame normalizer (isfinite before mask, ddof=0, versioned scale_floor)
     │   -> b[B,T,6] (FP32) + normalized fields [ũ,ṽ,w̃,p̃]
     │
     ├─ [ũ,ṽ,w̃,p̃, fluid_mask, SDF/D] -> 3D DC-AE Encoder (f8/C256, S2C residual shortcuts)
     │                               -> Spatial Transformer ×1 (8 heads, FFN 1024, PreLN, 3D pos)
     │                               -> Configurable Latent Norm (Default: GroupNorm + Global Scalar RMS;
     │                                  Ablation: token-wise affine-free RMSNorm)
     │                               -> z_clean [B,T,256,8,12,24]  (+ optional LatentPhysicsProbe)
     │
     └─ b[B,T,6] + geometry[B,2,Z,Y,X] ──────┐
                                             ▼
                                   Independent 3D Decoder (C2S ×3, no encoder-decoder skip)
                                   ├─ Head-A (Direct + Differentiable Helmholtz/Leray Projection or Div Penalty)
                                   └─ Head-B (Mean/Potential-Separated A* + Coulomb Gauge -> Compatible Curl_h)
                                   -> Physical Readout (with b and known fluid_mask) -> [û,v̂,ŵ,p̂]
     │
     ▼
G1: A/B head evaluation on held-out frames: valid-fluid reconstruction (m/s & Pa separate),
    first-layer near-wall error, interior divergence, vorticity & energy spectrum
     ▼
G2: 10-step lightweight latent predictor screening on validation split -> select & freeze Stage 1
    -> export SHA-256 bound cache (z_clean, b, conditions, dt_phys, dt_ctu, identity)
     ▼
Stage 2: history L=4 (z, b) [with training context noise σ_ctx & history-length dropout L_eff∈{1,2,4}]
         + 5 Condition Tokens [B,5,256]:
           e_text (Pluggable Structured-Physics / Frozen OpenCLIP)
           e_phys (log10 Re, U0, D, nu)
           e_step (Δt_phys AND convective Δt* = Δt·U0/D)
           e_action (known disturbance schedule only)
           e_regime (strictly inferred from history L=4 only)
     -> Causal Clean Track KV
     -> Query Track Dynamics (Two Clean Modes without Algebraic Shortcut):
        • Mode-FM (Stochastic Interpolant / Noisy-Source FM):
          z_0 = z_t + σ_0 ε (or noisy bridge z_τ = (1-τ)z_t + τ z_{t+1} + σ_b√(τ(1-τ))ε), K_ODE ∈ {1..8}
        • Mode-Res (K=1 Deterministic Residual / MeanFlow Baseline):
          ẑ_{t+1} = z_t + Δt* · v_θ(z_{t-3:t}, c) with 2..4 step pushforward rollout loss
     -> History-only b_hat head (positive scale parameterization + window-anchored damping option)
     -> (ẑ_{t+1}, b̂_{t+1}) -> Frozen Stage 1 Decoder -> FIFO Rollout
     ▼
G3: Held-out rollout evaluation (H=10/20/45 on 50-step trajectories; H=50/100 when >=105-step
    constant-dt trajectories are exported), separate u,v,w (m/s) & p (Pa) metrics, Strouhal phase,
    near-wall BC, spectral energy, condition counterfactuals, VRAM & throughput
```

---

## C. 关键设计隐患修正说明（为什么这样改）

1. **修正 Stage 2“无噪直线弦 Flow Matching”的代数恒等坍缩（P0 数学修正）**：
   - 原设计在训练时令 $z_\tau = (1-\tau)z_t + \tau z_{t+1}^{\text{true}}$ 且无随机噪声 $\varepsilon$，同时 $z_t$ 已在 `Clean Track` 历史中提供。对任意 $\tau \in (0, 1]$，目标速度存在平凡代数解 $v^* = z_{t+1}^{\text{true}} - z_t \equiv (z_\tau - z_t)/\tau$，导致网络可绕过物理学习直接拟合该恒等式，并在推理起点 $\tau=0, z_{\tau=0}=z_t$ 处遭遇 $0/0$ 奇异或退化为普通单步 $L_2$ 均值回归。
   - **修正后规范**：
     - **Mode-FM（随机桥 / 含噪源流匹配）**：训练与推理起点引入独立高斯噪声 $z_0 = z_t + \sigma_0 \varepsilon$（或随机桥 $z_\tau = (1-\tau)z_t + \tau z_{t+1}^{\text{true}} + \sigma_{\text{bridge}}\sqrt{\tau(1-\tau)}\,\varepsilon, \varepsilon \sim \mathcal{N}(0, I)$），彻底切断 $(z_\tau, z_t, \tau) \to z_{t+1}^{\text{true}}$ 的代数泄漏；
     - **Mode-Res（$K=1$ 确定性残差 / MeanFlow 基线）**：直接预测时间增量 $\hat{z}_{t+1} = z_t + \Delta t^* \cdot v_\theta(z_{t-3:t}, c)$ 并配合多步展开（Pushforward）损失，不做伪多步 ODE 循环。
2. **修正多连通开放边界域上纯旋度头 $\tilde{\mathbf{u}} = \nabla \times \mathbf{A}^*$ 的 Hodge–Helmholtz 不完备性（P0 物理修正）**：
   - 在含贯穿圆柱障碍物（第一 Betti 数 $b_1 = 1$）与开放进出口的有界三维域上，Hodge–Helmholtz 分解为 $\mathbf{u} = \nabla \times \mathbf{A} + \nabla \phi_{\text{harm}} + \mathbf{h}_{\text{circ}}$。纯局部旋度算子无法完备表达圆柱近壁势流偏转 $\nabla \phi_{\text{harm}} - \boldsymbol{\mu}$ 与环量模态，且同位中心差分 `curl_h` 在圆柱边界 1–2 层截断处并不满足离散无散，还存在奇偶棋盘格零空间与规范冗余 $\mathbf{A}^* \mapsto \mathbf{A}^* + \nabla \chi$。
   - **修正后规范**：在 R2/R3 中将 **Head-A（直接四场 / 势流基流残差 + 可微离散 Helmholtz/Leray 投影或软散度约束）** 与 **Head-B（均值/势流分离 + 库仑规范正则 $\|\nabla \cdot \mathbf{A}^*\|^2$ 的相容离散 `Curl` 头）** 列为同级 A/B 候选，在 G1 凭真实 C04 近壁第一层误差、内部散度与能谱择优冻结。
3. **修正 Token 级无仿射 RMSNorm 对空间湍动能对比的抹杀（P1 架构修正）**：
   - 保持 `f8 / C=256`（$8\times 12\times 24 = 2,304$ tokens）以保障单卡 3D Spatial Transformer 显存可行，靠 S2C/C2S 残差捷径 + 空间梯度/涡量损失保留剪切层高频；
   - 潜空间瓶颈默认采用 **`GroupNorm` + 数据集级全局标量 RMS 缩放**（遵循 MIT Han Lab DC-AE Appendix D），将“逐空间位置无仿射 RMSNorm”作为 `stage1.yaml` 中可显式开关的消融项（`latent_norm: group_norm | dataset_rms | affine_free_rmsnorm`）。
4. **修正逐帧 `b[6]` 自回归漂移、曝光偏差与跨雷诺数时间步尺度（P1 动力学修正）**：
   - `e_step` 同时编码物理步长 $\Delta t_{\text{phys}}$ 与**无量纲对流时间步长 $\Delta t^* = \Delta t \cdot U_0 / D$**（对齐多雷诺数 HDF5 数据集中恒定的 $\Delta t^* \approx 0.10\,\text{CTU}$）；
   - Stage 2 训练时对 `Clean Track` 历史潜变量注入微噪 $\sigma_{\text{ctx}} \in [0.01, 0.05]$ 并启用 **History-Length Dropout（$L_{\text{eff}} \in \{1, 2, 4\}$）**，防止历史捷径忽略五槽位条件与长滚曝光发散；
   - `b_hat` 预测头支持**历史窗口锚定阻尼更新（Window-Anchored Damping）**，抑制压力尺度 $\sigma_p$ 在 50 步自由滚动中的乘性漂移。

---

## D. 张量与数据契约（统一规范）

| 字段 | 形状 / 类型 | 语义与硬约束 |
| :--- | :--- | :--- |
| `field` | `[B, T, 4, Z, Y, X]` (`float32`) | 通道顺序固定 `[u, v, w, p]`；首版 `Z=64, Y=96, X=192`；$u$ 沿 $X$（`-1`）、$v$ 沿 $Y$（`-2`）、$w$ 沿 $Z$（`-3`） |
| `geometry` | `[B, 2, Z, Y, X]` (`float32`) | 通道固定 `[fluid_mask, sdf_over_d]`；`fluid_mask`: `1.0=fluid, 0.0=solid`；`sdf_over_d = SDF_m / D_m`（流体为正，壁面为 0，固体为负） |
| `spacing` | `(dz, dy, dx)` (`float32`) | 与 `(Z, Y, X)` 轴严格对应；C04 有量纲网格为 `(0.01875, 0.01875, 0.01875) m`，无量纲网格为 `(1/16, 1/16, 1/16)` |
| `b` | `[B, T, 6]` (`float32`) | `[mu_u, mu_v, mu_w, mu_p, sigma_uvw, sigma_p]`；在 FP64 下基于有效流体域 `M` 按 **population variance (`ddof=0`)** 计算后存为 FP32 |
| `z_clean` | `[B, T, 256, 8, 12, 24]` (`float32`) | Stage 1 Encoder + `Spatial Transformer ×1` 的干净潜表示（无时间 Transformer） |
| `history_z` | `[B, 4, 2304, 256]` (`float32`) | $L=4$ 历史潜网格按 `Z, Y, X` 展平为 $2304$ 个空间 Token |
| `conditions` | 强类型 `Flow3DConditions` | 含 `Re, u0_m_s, diameter_m, viscosity_m2_s, dt_phys_s, dt_ctu, action_schedule, text_prompt` 及 `known_mask / valid_mask` |
| `condition_tokens` | `[B, 5, 256]` (`float32`) | `[e_text, e_phys, e_step, e_action, e_regime]`；严禁装入任何未来真值或未来涡脱落阶段标签 |
| `future_stat` | `[B, 6]` (`float32`) | $\hat{b}_{t+1}$，仅由历史与已知条件预测；两个尺度分量通过 `softplus + scale_floor` 保证严格为正 |
| `identity` | 强类型 `SampleProvenance` | 统一全集：`case_id, geometry_id, trajectory_id, ancestor_id, segment_id, time_idx, time_phys_s, time_ctu, ground_truth_verified, tier1_stability_status` |

### 逐帧有效流体域归一化协议（`normalizer.flow3d.v1`）

在每帧有效流体掩码 $M = (\text{fluid\_mask} > 0.5)$ 上（先检验 `isfinite`！）：
1. **前置有限性硬拦截**：
   - 若 `fluid_mask` 或 `sdf_over_d` 含 `NaN/Inf`，直接抛出 `ValueError`；
   - 若有效流体点数 $|M| = 0$，直接抛出 `ValueError`（fail closed）；
   - 若任一物理通道 $c \in \{u, v, w, p\}$ **在有效流体域 $M$ 内**含 `NaN/Inf`，直接抛出 `ValueError`；
   - 固体域 $\neg M$ 内的 `NaN/Inf`（如 `video_c04_shedding` 中的 12,800 个固体 `NaN`）必须在索引 $M$ 之外被安全清洗为 `0.0`，**严禁使用 `values * mask`（因为 `NaN * 0.0 = NaN` 会污染全局统计）**。
2. **总体方差（`ddof=0`）与联合速度尺度**：
   $$\mu_c = \frac{1}{|M|} \sum_{x \in M} q_c(x), \quad c \in \{u, v, w, p\}$$
   $$\sigma_{uvw} = \max\!\left(\sqrt{\frac{1}{|M|} \sum_{x \in M} \left[(u - \mu_u)^2 + (v - \mu_v)^2 + (w - \mu_w)^2\right]},\ \text{scale\_floor}_{uvw}\right) \quad (\text{不除以 } 3)$$
   $$\sigma_p = \max\!\left(\sqrt{\frac{1}{|M|} \sum_{x \in M} (p - \mu_p)^2},\ \text{scale\_floor}_p\right)$$
   - `scale_floor` 版本化写入配置与 `NormalizationStats`；`fit()` 返回的 $\sigma_{uvw}, \sigma_p$ 已完成 `scale_floor` 截断，绝不返回 `0.0`。
3. **物理参数一致性硬校验**：
   - 当 `Re` 与 `(u0_m_s, diameter_m, viscosity_m2_s)` 同时存在时，强制校验：
     $$\frac{|Re - |U_0| D / \nu|}{Re} \le \epsilon_{\text{Re}} \quad (\text{默认 } \epsilon_{\text{Re}} = 10^{-2})$$
     超出容差（如 `re180k` 实际求解 $U_0=0.5\,\text{m/s} \implies Re=126167$ 或 `c04_wm_25steps_golden.h5` 属性与源模型 `Re=5001` 冲突）必须拒绝载入或自动打上 `QUARANTINED` 标签。

---

## E. 代码目录规划（`HydroDisturb-WorldModel` 路线 A 完整结构）

```text
HydroDisturb-WorldModel/
  docs/
    FlowWorld3D_V1_Development_Design.md
    FlowWorld3D_AI_Coding_R1_Prompt.md    # [已完成（f5dc000）] R1 / G0 提示词
    FlowWorld3D_AI_Coding_R2_Prompt.md    # R2 / G1 Stage 1 结构提示词（云端 GPU 下一阶段）
    g0_inspection_report.json             # [已完成（f5dc000）] 本机 3 套 3D 数据实测 G0 诊断报告
  src/
    stage1/                     # 既有 V1 基础层（保留原子 io/checkpoint.py 及既有 16+ 单测兼容）
    flow3d/                     # V1.3 正式三维主命名空间（按 R1..R5 分轮增量创建，禁止提交空文件）
      __init__.py               # [已完成（f5dc000）]
      contracts.py              # [已完成（f5dc000）] StateSpec3D, CYLINDER3D_STATE_SPEC, Flow3DBatch, 条件与来源强类型契约
      data/                     # [已完成（f5dc000）]
        schema.py               # [已完成（f5dc000）] ZYX<->XYZ 映射、dx/dy/dz、SDF/D、SampleProvenance、Re 一致性校验
        inspect.py              # [已完成（f5dc000）] 真实 3D 数据探针（支持 C04 NPZ / shedding NPZ 目录 / Multi-Re HDF5 + 伪数据隔离）
        adapters.py             # [已完成（f5dc000）] 显式格式载入器（处理 (96,64,192) 转置、w_cross/v_span 映射、h5 通道重排、固体 NaN 清洗）
        splits.py               # [已完成（f5dc000）] ancestor_id/trajectory_id 优先分组 + 单轨迹连续时间段（带 purge gap）双模式防泄漏切分
        normalization.py        # [已完成（f5dc000）] 逐帧有效域 6 维统计 b（ddof=0、前置 isfinite、版本化 scale_floor、roundtrip）
        fixtures.py             # [已完成（f5dc000）] 3D 解析无散场（Taylor-Green / 3D 旋度势生成场）与小体积合成 fixture
      physics/                  # [已完成（f5dc000）]
        operators.py            # [已完成（f5dc000）] (Z,Y,X) 轴对齐的 centered_difference_3d、div_h、curl_h 及离散 div(curl_h)=0 相容性检验
        boundary.py             # [已完成（f5dc000）] 圆柱近壁第一层流体点提取、法向/切向速度评估、进出口 BC 校验（拒绝把固体清零当无滑移）
      models/                   # [R2..R5 实现]
        __init__.py             # [R2 实现] 可选 PyTorch 导出入口
        dc_ae3d.py              # [R2 实现] 3D Stem + S2C/C2S 残差块 + 多尺度几何注入 + 独立 Decoder
        spatial_transformer.py  # [R2 实现] Spatial Transformer ×1 (8 heads, FFN 1024, PreLN, 3D SinCos/RoPE)
        heads.py                # [R2 实现] Head-A (Direct + Helmholtz/Leray Proj) & Head-B (Regularized A* -> Curl_h) + PhysicalReadout3D
        probe.py                # [R2 实现] LatentPhysicsProbe（可选消融：无量纲拟能与能谱探针）
        conditions.py           # [R4 实现] 5 槽位编码器（可插拔 StructuredPhysicsText / Frozen OpenCLIP + e_phys/e_step/e_action/e_regime）
        dynamics.py             # [R5 实现] Clean/Query 时空因果 Transformer（Mode-FM 含噪桥 & Mode-Res 一步残差 + history-only b_hat 头）
      training/                 # [R3..R5 实现]
        stage1.py
        stage2.py
      runtime/                  # [R3..R5 实现，直接复用/封装 stage1.io.checkpoint]
        checkpoint.py
        cache.py
      metrics/                  # [R3..R5 实现]
        reconstruction.py       # 严格分开报告速度 (m/s) 与压力 (Pa) 的 MAE/RMSE/Rel-L2、近壁误差、散度与能谱
        rollout.py
      cli/
        inspect_data.py         # [已完成（f5dc000）] 命令行数据探针与 G0 诊断报告导出
        train_stage1.py         # [R3 实现]
        validate_stage1.py      # [R3 实现]
        build_cache.py          # [R3 实现]
        train_stage2.py         # [R5 实现]
        evaluate_rollout.py     # [R5 实现]
  configs/
    stage1.yaml                 # 既有配置（保留）
    flow3d/
      data.yaml                 # [已完成（f5dc000）] 三维网格、物理常数、3 套本地数据源路径与隔离规则
      data.cloud.example.yaml   # 云端 Linux 数据路径模板与多雷诺数 .h5 上传/隔离白名单说明
      stage1.yaml               # [R2/R3 实现]
      stage2.yaml               # [R4/R5 实现]
  tests/
    test_*.py                   # 既有 6 个测试文件（必须保持 100% 通过）
    flow3d/
      test_contracts.py         # [已完成（f5dc000）]
      test_normalization.py     # [已完成（f5dc000）]
      test_axes_and_geometry.py # [已完成（f5dc000）]
      test_physics_operators.py # [已完成（f5dc000）]
      test_stage1_models.py     # [R2 实现]
  scripts/
    smoke_stage1.py             # 存量 Stage 1 烟雾脚本
    bench_stage1_flow3d.py      # [R2 实现] 64×96×192 Head-A/B 显存与吞吐基准
```

---

## F. 分轮 AI Coding 计划与验收门槛

| 轮次 | 状态 | 范围 | 关键输出 | 必须通过的验收门槛 |
| :--- | :--- | :--- | :--- | :--- |
| **R1 · G0** | **已完成（`f5dc000`）** | `src/flow3d/` 契约、数据探针、3 套真实数据源适配器、防泄漏切分、逐帧 6 维归一化、相容 `div_h/curl_h` 与真实壁面 BC、`cli/inspect_data.py` | 本机 3 套 3D 数据实测 G0 报告（`docs/g0_inspection_report.json`，含自动隔离 `c04_wm_25steps_golden.h5` 与 `re180k`）；`tests/flow3d/` 4 套单测（**云端 Agent 严禁重复实现 R1**） | 归一化往返误差 $\le 10^{-5}$；有效内部域 `div_h(curl_h(A)) == 0` 解析测试通过；固体 `NaN` 不污染统计；流体 `NaN` 与 $Re$ 错配必拦截；既有 `tests/test_*.py` 零回归 |
| **R2 · G1 结构** | **下一阶段（云端 GPU）** | 3D DC-AE (`S2C/C2S`) + `Spatial Transformer ×1` + 可配置 Latent Norm + Head-A (Direct/Proj) & Head-B (Regularized Curl) + `PhysicalReadout3D` | 前后向形状、梯度反传、Head-A/B 对照测试与 `64×96×192` 显存/吞吐基准（见 `docs/FlowWorld3D_AI_Coding_R2_Prompt.md`） | 编码器-解码器零跨层 skip；`Spatial Transformer ×1` 真实生效（禁 `depth=0`）；2 通道 `geometry` 解码无误；Head-A/B 在解析无散场上通过物理算子核验 |
| **R3 · G1/G2 训练** | 待执行 | Stage 1 训练闭环、原子 Checkpoint 恢复、Head-A/B 近壁与散度对比、10 步轻量潜预测筛选、SHA-256 冻结缓存导出 | 留出帧重构报告（速度 $\text{m/s}$ 与压力 $\text{Pa}$ 分列）、近壁第一层剖面、冻结 `z_clean + b` 缓存 | 测试集严禁参与选模型；缓存 manifest 绑定权重/契约/数据三项 SHA-256 |
| **R4 · 条件接口** | 待执行 | 五槽位 `condition_tokens [B,5,256]`（可插拔文本 + 物理 + 含 $\Delta t^*$ 步长 + 已知动作 + 纯历史流态） | 匹配/错配/恒定文本与物理参数反事实单测、因果掩码单测 | 未来不可见字段零泄漏；$Re \approx |U_0|D/\nu$ 冲突硬拦截；History-Length Dropout 验证生效 |
| **R5 · G3 演化** | 待执行 | Clean/Query 时空因果 Transformer（Mode-FM 含噪桥 & Mode-Res 一步残差）+ 历史专用正尺度 `b_hat` 头 + FIFO 自回归滚动 | 在 50 步真实轨迹上完成 $H=10/20/45$ 步（及长轨迹 $H=50/100$ 步）自由滚动评测，与 `Persistence` / `Delta-z` 基线对照 | 彻底消除 $(z_\tau - z_t)/\tau$ 代数捷径；分通道报告速度（$\text{m/s}$）与压力（$\text{Pa}$）误差、尾流相位/频率及散度 |

---

## G. 本仓库标准验证命令（路线 A）

### 1. 既有基线测试（任何轮次均不可回归）
```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_config.py tests/test_normalization.py tests/test_geometry_physics.py tests/test_npz_dataset.py tests/test_manifest.py tests/test_checkpoint.py -q
```

### 2. R1（G0）已完成验证命令（本地纯 NumPy 环境）
```powershell
python -m pytest tests/flow3d/test_contracts.py tests/flow3d/test_normalization.py -q
python -m pytest tests/flow3d/test_axes_and_geometry.py tests/flow3d/test_physics_operators.py -q
python -m pytest tests/flow3d -q
python -m pytest tests -q
python -m src.flow3d.cli.inspect_data --help
```

### 3. 云端 GPU 环境安装与 R2（G1 结构）验证命令
```bash
python -m pip install -e ".[all]"
python -m pytest tests -q
python -m pytest tests/flow3d/test_stage1_models.py -q
python scripts/bench_stage1_flow3d.py --config configs/flow3d/stage1.yaml --device cuda --dtype bf16
```

---

## H. 公开方法与文献依据

1. **DC-AE (MIT Han Lab, ICLR 2025)**: Chen et al., *Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models*, https://arxiv.org/abs/2410.10733 （S2C/C2S 残差高压缩比块及 Appendix D 数据集级全局潜空间 RMS 缩放）
2. **多连通域旋度势与势流基流分离**: Mohan et al. (2020), *Embedding Hard Physical Constraints in Neural Network Coarse-Graining of 3D Turbulence*, https://arxiv.org/abs/2002.00021 ; Sun (2026), *Learning in the Transverse Subspace*, https://arxiv.org/abs/2609.35884 ; Genuist et al. (2026), *Divergence-Free Diffusion Models for Incompressible Fluid Flows*, https://arxiv.org/abs/2601.19368
3. **湍流长程滚动稳定性与含噪历史条件**: Lippe et al. (2023), *PDE-Refiner*, https://arxiv.org/abs/2308.05732 ; Kohl et al. (2023), *ACDM*, https://arxiv.org/abs/2309.01745 ; Chen et al. (2024), *Diffusion Forcing*, https://arxiv.org/abs/2407.01392
4. **流匹配非高斯源理论与一步 MeanFlow**: Lipman et al. (ICLR 2023), *Flow Matching for Generative Modeling*, https://openreview.net/pdf?id=PqvMRDCJT9t ; Zhou et al. (2026), *Flow Matching from Non-Gaussian Sources*, https://arxiv.org/abs/2611.26407 ; Yang & Xue (2026), *MeLISA*, https://arxiv.org/abs/2605.05540
5. **物理条件注入与防历史捷径坍缩**: Herde et al. (2024), *Poseidon*, https://arxiv.org/abs/2405.19101 ; Dai et al. (2026), *PhysDial*, https://arxiv.org/abs/2611.23498
