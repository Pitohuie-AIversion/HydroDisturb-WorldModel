# FlowWorld-3D V1.3（路线 A 修订版）：三维流场时空世界模型 AI Coding 工程设计

> **主仓库绑定（路线 A）**：`D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-WorldModel`（Git 远端：`Pitohuie-AIversion/HydroDisturb-WorldModel.git`，已完成 R1 / G0 基线提交：`f5dc000`，初始骨架提交：`15aa63b`），服务于同级基准仓 `D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-Bench`（C04 三维高雷诺数圆柱绕流基准）。
> **修订说明**：本版本对齐路线 A，消除原 `Design.md` 与 `R1_Prompt.md` 的 8 处内部口径漂移，接入本机实测的 3 套三维 CFD 数据源契约，并修正了原路线中的 6 项数学、物理与自回归动力学隐患。

---

## A. 仓库现状与路线 A 最小变更策略

1. **主仓与存量模块处置**：
   - 本仓库 `src/stage1/` 已包含第一版协议骨架与**已验证的成熟原子检查点/冻结清单模块 `src/stage1/io/checkpoint.py`**（含临时文件落盘、检查点独占发布、SHA-256 指针、Python/NumPy/Torch CPU+CUDA RNG 恢复与 `CheckpointContract` 强校验），以及 6 个单元测试文件（`tests/test_*.py`）。
   - `src/stage1/` 中既有的 3 项 P0 数值缺陷（`physics/operators.py` 与 `torch_operators.py` 的 `u/w` 微分轴反转、`data/normalization.py` 固体区 `NaN * 0` 污染与 `fit()` 零尺度返回、`models/dcae.py` 未监督预测掩码门控与 2 通道几何切片崩溃）已先行修复，确保现有 `tests/test_*.py` 100% 绿色通过。
   - 存量 `src/stage1/models/dcae.py` 与 `src/stage1/training.py` 保留为 Stage 1 原型兼容层（不在 R1 中继续扩写）；正式 V1.3 体系在 **`src/flow3d/`、`configs/flow3d/`、`tests/flow3d/`** 独立命名空间下按 R1–R5 分轮演进，并直接复用 `src/stage1/io/checkpoint.py` 的原子持久化底座。
2. **本机实测三维 CFD 数据源现状（G0 必须显式适配与校验）**：
   - **数据源 1（C04 基准 4 帧体场序列）**：`D:\📁 01_Workspaces (代码与工程区)\HydroDisturb-Bench\runs\c04_s1_volume_sequence.npz`
     - 形状与通道：`fields (4, 4, 64, 96, 192) float32`，物理量纲 `[u, v, w, p]`（$\text{m/s}, \text{Pa}$），`times [0.0, 0.06, 0.12, 0.18] s`（$\Delta t = 0.06\,\text{s}$），`cell_spacing_m = 0.01875`，`cylinder_diameter_m = 0.3`；
     - 几何注意：`solid_mask (64, 96, 192) bool` 为 **`True = solid`（必须翻转为 `fluid_mask = 1 - solid_mask`）**；`sdf` 为**米制** `[-0.1367, 2.9704] m`（必须除以 $D = 0.3\,\text{m}$ 转换为无量纲 `SDF/D`）；
     - G0 用受信来源提供的 SHA-256 对照文件，并核对 C04 元数据、网格与时间。缺少独立摘要时报告 `UNVERIFIED`。
     - 窗口限制：仅 4 帧，可用于 G0/G1 单帧重构与算子核验，但**不足以构造 $L=4, H \ge 1$ 的 Stage 2 窗口**。
   - **数据源 2（C04 脱落过程 82 帧原始体场快照）**：`Z:\E\cfd_3d\05_reports_viz\video_c04_shedding\c04_vol_step_*.npz`（82 个文件，$t \in [0.0, 60.0]\,\text{s}$）
     - 轴序与通道映射：原始数组形状为 **`(96, 64, 192)` = `(Y_cross=96, Z_span=64, X_stream=192)`**（前两轴相对标准 `(Z=64, Y=96, X=192)` 转置！），且键名为 `u`（流向 $X$）、`w_cross`（横向 $Y$）、`v_span`（展向 $Z$）、`p`。适配器必须执行 `transpose(1, 0, 2)` 并将通道映射为 `[u, w_cross, v_span, p]`；
     - 固体区 `NaN` 红线：圆柱固体内部包含 **12,800 个 `NaN`（`finite=False`）**，必须在掩码外先行清洗并在有效流体域上先做 `isfinite` 校验；
     - 非均匀时间步红线：82 帧跨度内 $\Delta t$ **非均匀**（含重复时间戳 `0.0s` 1 处、`0.12s` 25 处、`0.24s` 21 处、`0.96s` 6 处、`1.02s` 10 处、`1.98s` 12 处、`2.04s` 6 处），窗口索引器必须按**连续恒定 $\Delta t$ 子段（Segment）**切分，严禁跨变步长跳变点拼接固定步长窗口。
   - **数据源 3（多雷诺数 HDF5 世界模型数据集）**：`Z:\E\cfd_3d\03_world_model_dataset\*.h5` 及 `manifest.json`、`multi_re_tier1_certification_summary.json`
     - 包含 3 条已通过真值核验的 50 步无量纲轨迹：`re200k_wm_50steps.h5`（$Re=2\times 10^5, \Delta t=0.041\,\text{s}, \Delta t^*=0.108\,\text{CTU}$）、`re350k_wm_50steps.h5`（$Re=3.5\times 10^5, \Delta t=0.02\,\text{s}, \Delta t^*=0.092\,\text{CTU}$）、`re1m_wm_50steps.h5`（$Re=10^6, \Delta t=0.01\,\text{s}, \Delta t^*=0.132\,\text{CTU}$），形状均为 `(50, 6, 64, 96, 192)`；
     - 通道顺序与极性红线：HDF5 内置 6 通道顺序为 `["u_star", "v_cross_star", "w_span_star", "p_star", "sdf_star", "solid_mask"]`——**第 4 通道是 `sdf_star`、第 5 通道是 `solid_mask`（`1=solid`）**，适配器必须将第 4/5 通道重排并翻转为 `[fluid_mask = 1 - solid_mask, SDF/D = sdf_star]`；
     - **训练资格**依赖清单来源、文件签名、可读张量、物性、时间轴、通道和逐帧有效流体域。已知 `c04_wm_25steps_golden.h5` 的来源审计确认插值造假，`re180k` 的求解配置与标签冲突，两者隔离。对新增样本记录差分秩，并检查相对首尾帧的插值残差。低秩周期场可进入训练；端点线性匹配且缺少解释的轨迹转为 `UNVERIFIED`。仅能读取清单和文件签名时也记为 `UNVERIFIED`。

---

## B. 正式总体架构（消除数学与物理隐患后的 V1.3 定型版）

```text
CFD [u,v,w,p] + ZYX Grid (64×96×192) + Geometry [fluid_mask, SDF/D] + Provenance/Conditions
     │
     ▼
G0: inspect/validate axes (ZYX vs YZX), units, Re≈|U0|D/nu, solid NaNs, dt segments,
    audit source and tensor, check endpoint interpolation, split by common ancestor and trajectory
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
     -> Causal Clean Track KV (Factorized Space-Time Attention: 2304 spatial + L=4 temporal)
     -> Query Track Dynamics (Unmasked Cross-Attention to Clean-KV; Two Modes without Algebraic Shortcut or Endpoint Singularity):
        • Mode-FM (Bounded-Derivative Noisy-Source / Smooth-Bridge FM):
          z_0 = z_t + σ_0 ε, z_τ = (1-τ)z_0 + τ z_{t+1} (v* = z_{t+1} - z_0), OR smooth quadratic bridge
          γ(τ) = σ_b τ(1-τ) (γ̇(τ) = σ_b(1-2τ)) / decoupled (v, η) loss (FORBID direct ∂_τ√(τ(1-τ)) singularity!)
        • Mode-Res (deterministic residual dynamics, one update per CFD output step):
          Clean/Query uses the last observed increment plus Δt* times a learned correction.
          Both z and b roll forward under supervision at each training lead.
     -> History-only b_hat head (positive scale parameterization + window-anchored damping option)
     -> (ẑ_{t+1}, b̂_{t+1}) -> Frozen Stage 1 Decoder -> FIFO Rollout
     ▼
G3: held-out transient rollout H=10/20/45 or H=50/100 where a constant-step window fits.
    Score SI field errors against Persistence and Delta-z. Gate spectral scores on stable cycle coverage,
    and intervention scores on paired CFD branches from a shared history (acceptance criteria in section F).
```

---

## C. 关键设计隐患修正说明（为什么这样改）

1. **修正 Stage 2“无噪直线弦 Flow Matching”的代数恒等坍缩与棕色桥端点导数奇异性（P0 数学修正）**：
   - 原设计在训练时令 $z_\tau = (1-\tau)z_t + \tau z_{t+1}^{\text{true}}$ 且无随机噪声 $\varepsilon$，同时 $z_t$ 已在 `Clean Track` 历史中提供。对任意 $\tau \in (0, 1]$，目标速度存在平凡代数解 $v^* = z_{t+1}^{\text{true}} - z_t \equiv (z_\tau - z_t)/\tau$，导致网络可绕过物理学习直接拟合该恒等式，并在推理起点 $\tau=0, z_{\tau=0}=z_t$ 处遭遇 $0/0$ 奇异或退化为普通单步 $L_2$ 均值回归。
   - 同时，若采用棕色桥 $\gamma(\tau) = \sigma_b\sqrt{\tau(1-\tau)}$ 并直接对 $z_\tau$ 求导作为速度目标，$\dot{\gamma}(\tau) = \sigma_b\frac{1-2\tau}{2\sqrt{\tau(1-\tau)}}$ 在 $\tau \to 0^+, 1^-$ 时发散至 $\pm\infty$，会直接引发训练 Loss 方差爆炸与 `NaN`（参见 *Albergo et al.* `arXiv:2303.08797`, *SiT* `arXiv:2401.08740`）。
   - **修正后规范**：
     - **Mode-FM（有界导数含噪源 / 平滑随机桥流匹配）**：默认采用**含噪源直线流匹配** $z_0 = z_t + \sigma_0 \varepsilon,\ z_\tau = (1-\tau)z_0 + \tau z_{t+1}^{\text{true}}$（有界常速度目标 $v^* = z_{t+1}^{\text{true}} - z_0 = z_{t+1}^{\text{true}} - z_t - \sigma_0\varepsilon$），或采用**平滑二次桥** $\gamma(\tau) = \sigma_b \tau(1-\tau)$（有界导数 $\dot{\gamma}(\tau) = \sigma_b(1-2\tau)$）/ **解耦 $(v, \eta)$ 双目标损失**（严禁直接对 $\sqrt{\tau(1-\tau)}$ 求导回归）；
     - **Mode-Res（确定性残差动力学）**。当前 Clean/Query 在恒定时间步长段用最近两帧的潜变量增量作初值，再学习 $\Delta t^*\,v_\theta$ 修正。只保留一帧训练历史时，初始增量为零。本次训练对两步展开中的每步潜变量与背景量计损失。文献中的 MeanFlow 通过平均速度与瞬时速度的恒等式学习生成流，属于另一项训练目标。[MeanFlow 原论文](https://arxiv.org/abs/2505.13447)
2. **修正多连通开放边界域上纯旋度头 $\tilde{\mathbf{u}} = \nabla \times \mathbf{A}^*$ 的 Hodge–Helmholtz 不完备性、近壁死区与第二层壳散度爆炸（P0 物理修正）**：
   - 在含贯穿圆柱障碍物（第一 Betti 数 $b_1 = 1$）与开放进出口的有界三维域上，Hodge–Helmholtz 分解为 $\mathbf{u} = \nabla \times \mathbf{A} + \nabla \phi_{\text{harm}} + \mathbf{h}_{\text{circ}}$。
   - **三项实测关键修正**：
     1. **潜势求导传 `mask=None`**：因为 $\mathbf{A}^*$ 是神经网络自由预测的连续潜势（不含 CFD 固体 `NaN`），在 `curl_h_torch(A*, spacing_zyx, mask=None)` 中**不得传入 `fluid_mask` 截断导数**——否则不仅会把 100% 圆柱第一层流体壳（`3,072` 点）与 6 个外边界面（`71,912` 点，合计 `6.42%` 流体域）清零成常数 $\mathbf{u}_{\text{mean}}$，还会在圆柱第二层流体壳引发高达 $403\text{–}4481$ 的阶跃截断伪散度；传入 `mask=None` 后该处离散散度直接降至 $6.1\times 10^{-5}$（降幅 660 万倍），物理场仅在最终 `PhysicalReadout3D` 固体域内按 `fluid_mask` 掩码！
     2. **势流分支的调和约束**。令 `u_pot = grad_h(phi)`，将这一速度分量同时用于输出与 `div_h(u_pot)`。`harmonic_laplacian_loss` 在双层完整流体模板内逐帧计算散度均方，再对帧取平均。中心差分梯度后接中心差分散度的模板跨两个网格单元；独立的七点拉普拉斯对同一离散场可以给出不同结果。`coulomb_gauge_loss` 使用一层完整模板。
     3. **Head-A 禁用周期 FFT 投影，主用 `soft_penalty`**：浸没圆柱与开放进出口域不满足周期对角 Fourier 投影条件（`arXiv:2610.07811`, `arXiv:2601.19368`），且在训练图内朴素展开多步 3D 掩码 CG 会爆显存并在 BF16/FP16 下发散；本次 A830 的训练损失含有效流体区的速度与压力重构项，以及权重 0.01 的散度罚项。第一层流体壳速度用于评估。独立的近壁损失和有限步投影属于可选实验，A830 的投影开关为关闭。
3. **修正 Stage 2 注意力架构、小样本训练策略与云端环境/数据加载规范（P0/P1 工程修正）**：
   - **Stage 2 注意力预算**。Clean Track 使用分解的时空注意力，Query Track 对历史 KV 使用无掩码交叉注意力。R5 按目标 GPU、dtype、序列长度和批量记录实际选用的 SDPA 后端，并在前向与反向实测峰值显存。
   - **单位换算**。HDF5 入库时使用文件与清单核对过的参考量，按 `u = U0 · u*`、`p = p∞ + ρ0 U0² · p*` 转为 SI；C04 NPZ 已使用 SI。两条路径都对 SI 场计算 `b`，逆归一化直接得到 m/s 与 Pa。物理诊断用 `batch.spacing`（m），归一化场的物理头用 `batch.spacing_over_d(i)`。不同无量纲网格间距的样本分别调用物理头。
   - **采样截面与物理边界**。C04 求解域的流向范围为 [−3, 6] m，采样范围为 [−0.6, 3] m。速度、压力在截面处与同位置 CFD 比较。用最靠近六个采样面的格心速度乘以网格面面积，按向外为正累加，得到局部区域的近似净流量（m³/s）。圆柱近壁指标单独报告。[上游 C04 配置](https://github.com/Pitohuie-AIversion/HydroDisturb-Bench/blob/5ac9cf5c2df9da9e8cc38a0ea91a030f140032c0/configs/c04_re126k_3d_baseline.json#L25-L88)
   - **训练与评测数据**。当前三条 50 帧多 Re 轨迹可用于 R2 结构验证、R3 原型及瞬态预测诊断。三条的 Tier-1 均为 `UNSTABLE`。R3 可使用 `32×48×64` 随机子块、展向镜像及 `SDF/D` 裁剪至 `[-8,8]`。同一祖先的所有增强样本随原轨迹划分。跨工况采集目标保留为 6–10 条独立轨迹、每条 105–200 步；对频率评测按 F 节的周期数和目标分辨率延长稳态段，对干预评测补充成对分支。
   - **云端 PyTorch / HDF5 硬护栏**：① 锁定 **PyTorch `2.5.1–2.8.0`**（避开 PyTorch `2.9.0/2.9.1+cu128/cu129` 的 `Conv3d` BF16 cuDNN 17.5× 显存暴涨与卡顿回归 `pytorch#166643/#166790/#168167`）；② 锁定 **`h5py>=3.14`**（兼容 `numpy>=2.0` 并包含 Linux `phil` 锁 fork 修复 `h5py#2584`），且多进程 `DataLoader` 强制 `multiprocessing_context="spawn"` + worker 内懒打开/即关 `h5py.File`；③ `src/stage1/io/checkpoint.py` 恢复 RNG 时强制 `.cpu()` 并容忍单/多卡设备数变化。
3. **修正 Token 级无仿射 RMSNorm 对空间湍动能对比的抹杀（P1 架构修正）**：
   - 保持 `f8 / C=256`（$8\times 12\times 24 = 2,304$ tokens）以保障单卡 3D Spatial Transformer 显存可行，靠 S2C/C2S 残差捷径 + 空间梯度/涡量损失保留剪切层高频；
   - 潜空间瓶颈默认采用 **`GroupNorm` + 数据集级全局标量 RMS 缩放**（遵循 MIT Han Lab DC-AE Appendix D），将“逐空间位置无仿射 RMSNorm”作为 `stage1.yaml` 中可显式开关的消融项（`latent_norm: group_norm | dataset_rms | affine_free_rmsnorm`）。`latent.flow3d.v2` 在空间 Transformer 末端直接输出残差流。三组消融共享参数初值、样本和输入，分别测量逐位置通道 RMS 的空间分布及留出集重构误差。已有 v1 权重复现时保留其末端 LayerNorm，并使用原潜变量契约和缓存。
4. **修正逐帧 `b[6]` 自回归漂移、曝光偏差与跨雷诺数时间步尺度（P1 动力学修正）**：
   - `e_step` 同时编码物理步长 $\Delta t_{\text{phys}}$ 与**无量纲对流时间步长 $\Delta t^* = \Delta t \cdot U_0 / D$**（对齐多雷诺数 HDF5 数据集中恒定的 $\Delta t^* \approx 0.10\,\text{CTU}$）；
   - Stage 2 训练时对 `Clean Track` 历史潜变量注入微噪 $\sigma_{\text{ctx}} \in [0.01, 0.05]$ 并启用 **History-Length Dropout（$L_{\text{eff}} \in \{1, 2, 4\}$）**，防止历史捷径忽略五槽位条件与长滚曝光发散；
   - `b_hat` 预测头支持**历史窗口锚定阻尼更新（Window-Anchored Damping）**，抑制压力尺度 $\sigma_p$ 在 50 步自由滚动中的乘性漂移。

---

## D. 张量与数据契约（统一规范）

| 字段 | 形状 / 类型 | 语义与硬约束 |
| :--- | :--- | :--- |
| `field` | `[B,T,4,Z,Y,X]` (`float32`) | `[u,v,w,p]`，速度单位 m/s，压力单位 Pa。`u,v,w` 分别沿最后三维中的 X、Y、Z 轴。 |
| `geometry` | `[B, 2, Z, Y, X]` (`float32`) | 通道固定 `[fluid_mask, sdf_over_d]`；`fluid_mask`: `1.0=fluid, 0.0=solid`；`sdf_over_d = SDF_m / D_m`（流体为正，壁面为 0，固体为负） |
| `spacing` | `(dz,dy,dx)` | SI 网格间距，单位 m。`spacing_over_d(i)` 返回样本 i 的无量纲间距，C04 对应 `(1/16,1/16,1/16)`。 |
| `b` | `[B,T,6]` (`float32`) | `[mu_u,mu_v,mu_w,mu_p,sigma_uvw,sigma_p]`，速度项为 m/s，压力项为 Pa。有效流体域内以 FP64 计算总体方差后存为 FP32。 |
| `z_clean` | `[B, T, 256, 8, 12, 24]` (`float32`) | Stage 1 Encoder + `Spatial Transformer ×1` 的干净潜表示（无时间 Transformer） |
| `history_z` | `[B, 4, 2304, 256]` (`float32`) | $L=4$ 历史潜网格按 `Z, Y, X` 展平为 $2304$ 个空间 Token |
| `conditions` | 强类型 `Flow3DConditions` | 含 `Re, u0_m_s, diameter_m, viscosity_m2_s, dt_phys_s, dt_ctu, action_schedule, text_prompt` 及 `known_mask / valid_mask` |
| `condition_tokens` | `[B, 5, 256]` (`float32`) | `[e_text, e_phys, e_step, e_action, e_regime]`；严禁装入任何未来真值或未来涡脱落阶段标签 |
| `future_stat` | `[B, 6]` (`float32`) | $\hat{b}_{t+1}$，仅由历史与已知条件预测；两个尺度分量通过 `softplus + scale_floor` 保证严格为正 |
| `identity` | 强类型 `SampleProvenance` | 统一全集：`case_id, geometry_id, trajectory_id, ancestor_id, segment_id, time_idx, time_phys_s, time_ctu, ground_truth_verified, tier1_stability_status` |

### 逐帧有效流体域归一化协议（`normalizer.flow3d.si.v2`）

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
    FlowWorld3D_AI_Coding_R2_Prompt.md    # R2 / G1 Stage 1 结构提示词（云端 GPU 验证）
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
        boundary.py             # [已完成（f5dc000）] 圆柱第一层流体壳速度与采样体积六面向外流量（m³/s）
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
        rollout.py              # 逐帧 SI 集合均值 RMSE、速度能量评分与压力 CRPS
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
      test_encoder_decoder.py / test_spatial_transformer.py     # [R2 实现]
  scripts/
    smoke_stage1.py             # 存量 Stage 1 烟雾脚本
    smoke_flow3d_stage1.py      # [R2 实现] 64×96×192 Head-A/B 显存与吞吐基准
```

---

## F. 分轮 AI Coding 计划与验收门槛

| 轮次 | 状态 | 范围 | 关键输出 | 必须通过的验收门槛 |
| :--- | :--- | :--- | :--- | :--- |
| **R1 · G0** | **已完成（`f5dc000`）** | `src/flow3d/` 契约、数据探针、3 套真实数据源适配器、防泄漏切分、逐帧 6 维归一化、相容 `div_h/curl_h` 与真实壁面 BC、`cli/inspect_data.py` | 本机 3 套 3D 数据实测 G0 报告（`docs/g0_inspection_report.json`，含自动隔离 `c04_wm_25steps_golden.h5` 与 `re180k`）；`tests/flow3d/` 4 套单测（**云端 Agent 严禁重复实现 R1**） | FP32 归一化往返满足 R2 的逐通道 SI 容差；有效内部域 `div_h(curl_h(A)) == 0` 解析测试通过；固体 `NaN` 不污染统计；流体 `NaN` 与 $Re$ 错配必拦截；既有 `tests/test_*.py` 零回归 |
| **R2 · G1 结构** | 已通过结构验证 | 3D DC-AE (`S2C/C2S`) + `Spatial Transformer ×1` + 可配置 Latent Norm + Head-A (Direct/Proj) & Head-B (Regularized Curl) + `PhysicalReadout3D` | 前后向形状、梯度反传、Head-A/B 对照测试与 `64×96×192` 显存/吞吐基准（见 `docs/FlowWorld3D_AI_Coding_R2_Prompt.md`） | 编码器-解码器零跨层 skip；`Spatial Transformer ×1` 真实生效（禁 `depth=0`）；2 通道 `geometry` 解码无误；Head-A/B 在解析无散场上通过物理算子核验 |
| **R3 · G1/G2 训练** | 已完成双头重构比较和独立验证祖先十步筛选 | 重构训练与轻量潜预测筛选 | 按留出祖先分别报告 m/s 和 Pa 误差、同层近壁速度误差及散度 | 用验证集选模型。缓存 v2 验证 checkpoint、归一化与数据集身份、索引和实际分片的 SHA-256。 |
| **R4 · 条件接口** | 确定性五槽接口已运行，成对物理干预待评 | 物理条件、已知动作和纯历史流态 | 匹配、错配、恒定条件及未来不可见性测试 | 条件置换检验接口；干预的物理响应按下文的成对 CFD 分支验收。 |
| **R5 · G3 演化** | 确定性支路已在验证集运行，正式 G3 测试待评 | Clean/Query 动力学与 FIFO 自回归 | 同一留出集上对照 Persistence 和 Delta-z 的逐时距物理误差 | 瞬态、稳态频谱、干预响应各自满足下列条件后报告。 |

### G3 数据采集与通过条件

1. **瞬态预测**。在按祖先留出的同一恒定步长段内，对历史长度 `L`、预测长度 `H` 计数 `max(0, N−L−H+1)`。`N=50,L=4,H=45` 每条仅有两个高度重叠的起点。报告起点数和独立祖先数，误差先按祖先汇总。验证集先固定主时距和指标，测试集一次评分。点预测按有效流体格点分别计算速度向量 RMSE（m/s）与压力 RMSE（Pa），两项各自须优于 Persistence 和 Delta-z 中该项更优者。逐通道误差和物理残差另列。
   - **Mode-FM 集合**。验证前固定成员数（默认 16）、种子列表、积分器和步数。每个成员使用独立随机流从同一历史滚动至 H，在各时距先将每个成员独立解码到 SI，再对成员的物理场取均值计算点预测 RMSE。用 ODE 采样时，两种桥都从含独立噪声的源分布起步，记录源噪声尺度。
   - **概率预测**。`flow3d.metrics.rollout.score_forecast_frame_si` 接收一个时刻的 SI 成员场、观测和流体掩码。每个格点对三维速度使用能量评分，对标量压力使用 CRPS，各保留物理单位。对 K 个独立成员，取成员到观测的平均距离，再减去所有不同成员两两距离平均值的一半（速度用欧氏距离，压力用绝对值）。确定性基线取预测到观测的距离。先在有效流体域内平均，再按预先固定的时距、起点与留出祖先汇总。两项分数越低越好，各自与对应的最佳基线比较，单独报告概率预测是否通过。用等概率 −1/+1 真值回归检验，正确分布和恒零预测的总体期望 CRPS 应分别为 0.5 和 1。[有限集合的公平评分](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf) · [概率预报评分定义](https://stat.uw.edu/research/tech-reports/strictly-proper-scoring-rules-prediction-and-estimation-revised)
2. **稳态频率与相位**。采集时同步导出升力系数 `Cl(t)` 并标出通过 Tier-1 的稳态区间。预测场若通过观测算子提取尾流信号，则用相同位置、分量、去均值方式和相位零点提取对应 CFD 信号；在留出评测前固定观测量。每个评测窗口须位于稳态区间，覆盖至少五个在真值中实际测得的完整周期。直接频谱的频点间隔 `1/(H·Δt*)` 还须满足预先登记的目标。以 `ΔSt≤0.02`、`Δt*=0.1` 为例，需要至少 500 个预测采样点，再加历史帧。相位误差用相同探针的同向过零时刻，按真值周期折算并报告。当前三条轨迹只报告瞬态指标；G0 的 `g3_window_support` 记录各时距的起点数和频点间隔。
3. **扰动响应**。从同一求解器检查点导出至少两条分支，记录四帧共同历史、相同物性和后续时间轴。分叉时仅改变指定的已知外部动作，并保留两条 CFD 结果。清单中提供共同的 `ancestor_id`、`boundary_condition_id`、`pressure_gauge`，以及 `intervention` 内的 `pair_id`、`branch_id`、`channel`、`unit`、`exogenous: true`、`known_at_time_s`、`times_s`、`values`。动作序列与 HDF5 物理时间轴对齐，且在预测起点前已确定。G0 核对实际历史与几何、时间轴和动作差异后计数。整组分支进入同一数据划分，测试集使用独立留出祖先。Mode-FM 对同一对分支使用相同成员种子及逐步噪声，逐成员相减后取集合均值。逐工况报告预测响应 `ŷ_action−ŷ_reference` 与 CFD 响应 `y_action−y_reference` 的幅值、符号和误差。存在可评分的非零响应时，响应 RMSE 须优于零响应基线。没有合格对照组时报告可用对数 0，物理干预项标为待评。

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
python -m pytest tests/flow3d/test_encoder_decoder.py tests/flow3d/test_spatial_transformer.py -q
python scripts/smoke_flow3d_stage1.py --config configs/flow3d/stage1.yaml --device cuda:0 --amp
```

---

## H. 公开方法与文献依据

1. **压缩与潜空间演化。** Chen 等研究图像 DC-AE 的残差压缩，附录 D 讨论下游模型的潜变量缩放。Wiewel 等用压缩表示预测三维流体压力。项目实际使用的 f8/C256、潜变量尺度和重构误差由本地检查点与 CFD 验证确定。[DC-AE](https://arxiv.org/html/2410.10733v8#A4) · [Latent-space Physics](https://arxiv.org/abs/1802.10123)
2. **无散速度表示。** Mohan 等在三维湍流粗粒化网络中嵌入不可压约束。横向子空间的可辨识性和无散扩散模型也有专门研究。本项目 Head-B 的近壁误差、网格算子和有效内部域须各自检验。[Mohan 等](https://arxiv.org/abs/2002.00021) · [Sun](https://arxiv.org/abs/2609.35884) · [Genuist 等](https://arxiv.org/abs/2601.19368)
3. **自回归与长时距。** PDE-Refiner 研究空间频率对时间滚动的影响。Kohl 等分开评估预测精度、谱统计与时间稳定性。Diffusion Forcing 研究带不同噪声水平的序列建模。本项目已用验证轨迹测量 H10、H20、H45 的点预测误差。[PDE-Refiner](https://arxiv.org/abs/2308.05732) · [Kohl 等](https://arxiv.org/html/2309.01745v2) · [Diffusion Forcing](https://arxiv.org/abs/2407.01392)
4. **生成动力学。** Flow Matching 回归指定概率路径的向量场，MeanFlow 用平均速度与瞬时速度的关系训练一步生成器。非高斯源采样理论和一步生成式动力学预测可供未来的 Mode-FM 对照。已训练的 Mode-Res 使用监督式残差更新。[Flow Matching](https://arxiv.org/abs/2210.02747) · [MeanFlow](https://arxiv.org/abs/2505.13447) · [Zhou 等](https://arxiv.org/abs/2611.26407) · [Yang 与 Xue](https://arxiv.org/abs/2605.05540)
5. **物理条件与干预。** Poseidon 使用时间条件归一化。PhysDial 从同一求解器状态比较条件改变后的分支。项目的静态条件遮蔽仅测输入敏感性，真实扰动响应需要共同历史下的 CFD 分支。[Poseidon](https://arxiv.org/abs/2405.19101) · [PhysDial](https://arxiv.org/abs/2611.23498)

当前权重、逐工况基线和独立重放结果见[确定性预测的验证复核](FlowWorld3D_Validation_Calibration_20261124.md)。
