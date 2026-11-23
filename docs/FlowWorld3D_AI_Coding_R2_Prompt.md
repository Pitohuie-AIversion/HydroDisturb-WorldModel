# 给云端 GPU Coding AI 的第二轮任务：FlowWorld-3D V1.3（路线 A）· R2 / G1 Stage 1 结构实现

你在主仓库 `HydroDisturb-WorldModel`（Git 远端 `Pitohuie-AIversion/HydroDisturb-WorldModel.git`，已完成 R1/G0 的基准提交为 `f5dc000`）中执行本轮任务。完整工程设计见同目录 [`docs/FlowWorld3D_V1_Development_Design.md`](FlowWorld3D_V1_Development_Design.md)，首轮已完成基线见 [`docs/FlowWorld3D_AI_Coding_R1_Prompt.md`](FlowWorld3D_AI_Coding_R1_Prompt.md) 与 [`docs/g0_inspection_report.json`](g0_inspection_report.json)。

**本轮仅实现 R2 / G1 Stage 1 网络结构与物理读出层：3D DC-AE（S2C/C2S 残差块 + 多尺度几何注入 + 零跨层 skip 独立解码器）+ `Spatial Transformer ×1` + 可配置 Latent Norm + Head-A / Head-B 双候选物理头 + `PhysicalReadout3D` + 可选 `LatentPhysicsProbe` + `configs/flow3d/stage1.yaml` + R2 单元测试与 `64×96×192` 显存/吞吐基准。**

> **严格边界红线**：
> 1. **严禁重复实现 R1 / G0**：`src/flow3d/contracts.py`、`src/flow3d/data/*`、`src/flow3d/physics/*`、`src/flow3d/cli/inspect_data.py` 及 `tests/flow3d/test_{contracts,normalization,axes_and_geometry,physics_operators}.py` 已在 `f5dc000` 验收通过，本轮必须直接复用，不得重写或弱化既有契约。
> 2. **严禁破坏存量兼容层**：`src/stage1/`（尤其是 `src/stage1/io/checkpoint.py`）与既有 `tests/test_*.py` 必须保持 100% 绿色通过。
> 3. **严禁提前越界实现 R3/R4/R5**：本轮不写完整 Stage 1 训练闭环与 G2 筛选脚本（R3），不写五槽位条件编码器/CLIP（R4），不写 Stage 2 Clean/Query 时空动力学模型（R5）。

---

## 1. 云端 GPU 环境与数据准备（执行前核对）

1. **安装完整依赖（含 PyTorch、HDF5、PyYAML、Pytest）**：
   ```bash
   python -m pip install -e ".[all]"
   python -c "import torch, h5py, yaml; print(torch.__version__, torch.cuda.is_available())"
   ```
2. **云端数据配置与隔离红线**：
   - 如需在云端验证真实数据路径，请复制 [`configs/flow3d/data.cloud.example.yaml`](../configs/flow3d/data.cloud.example.yaml) 为 `configs/flow3d/data.cloud.yaml` 并修改 Linux 路径（不要覆盖本地 Windows 路径的 `configs/flow3d/data.yaml`）。
   - 仅允许使用 3 条经 G0 核验的多雷诺数 50 步轨迹：`re200k_wm_50steps.h5`、`re350k_wm_50steps.h5`、`re1m_wm_50steps.h5`（以及可选的 C04 4 帧基准 `c04_s1_volume_sequence.npz`）。
   - **严禁上传或加载被隔离的数据**：`c04_wm_25steps_golden.h5`（`ground_truth_verified: false`，`Re=5000` 秩-2 线性插值伪数据误标为 `Re=126167`）与 `re180k_mesh1_pilot`（求解器 $U_0=0.5\,\text{m/s}$ 导致真实 $Re=126167.05$ 与标称 $180000$ 冲突）。
3. **先运行既有测试确认基线全绿**：
   ```bash
   git status --short
   git rev-parse --short HEAD
   python -m pytest tests -q
   ```

---

## 2. 本轮具体实现范围（R2 / G1 Stage 1 结构）

在 `src/flow3d/models/`、`configs/flow3d/` 与 `tests/flow3d/` 下创建并实现以下模块（禁止提交空文件、占位 `pass` 或 `depth=0` 假层）：

### 2.1 `src/flow3d/models/__init__.py`
- 统一导出 R2 公共类与构建入口（含 `DC_AE3DConfig`、`Encoder3D`、`Decoder3D`、`Flow3DAutoencoder`、`SpatialTransformer3D`、`DirectProjectionHead3D`、`RegularizedCurlHead3D`、`PhysicalReadout3D`、`LatentPhysicsProbe`）。
- 遵循 `src/stage1/models/dcae.py` 的可选 PyTorch 导入风格：当环境未安装 `torch` 时模块仍可安全导入并在实例化时抛出清晰安装提示（`pip install -e ".[all]"`），确保纯 NumPy 环境下 R1 既有测试零回归。

### 2.2 `src/flow3d/models/dc_ae3d.py`（3D DC-AE 编解码主干）
严格对齐设计文档第 B、C.3、D 节：
1. **输入与空间压缩契约**：
   - 输入帧为 `[B, 6, Z, Y, X]`（或 `[B, T, 6, Z, Y, X]` 经 `B*T` 展平后逐帧编码），由归一化物理场 `[ũ, ṽ, w̃, p̃]`（4 通道）与规范几何 `[fluid_mask, SDF/D]`（2 通道）拼接而成；标准空间网格 `(Z=64, Y=96, X=192)`。
   - 3D Stem + 3 级 **S2C（Space-to-Channel）残差下采样块**（遵循 MIT Han Lab DC-AE ICLR 2025），总空间下采样倍率 **`f8`**，输出潜特征图 `[B, 256, Z/8, Y/8, X/8]`（标准网格为 `[B, 256, 8, 12, 24]`）。
   - 多尺度几何注入：在编码器各分辨率阶段注入下采样后的 2 通道 `geometry = [fluid_mask, SDF/D]`（`trilinear` 对 `SDF/D`、`area`/`nearest` 或一致下采样对 `fluid_mask`，严禁越界切片或硬编码 6 通道假设）。
2. **可配置潜空间归一化（Latent Norm）**：
   - 在 `Spatial Transformer ×1` 之后输出 `z_clean [B, 256, 8, 12, 24]`（或 `[B, T, 256, 8, 12, 24]`），支持通过配置切换三种模式：
     - `group_norm`（**默认**）：`GroupNorm` + 可注册 buffer 的数据集级全局标量 RMS 缩放（`latent_rms_scale`，默认 `1.0`，遵循 DC-AE Appendix D，保护空间湍动能对比不被抹杀）；
     - `dataset_rms`：仅执行数据集级全局标量 RMS 缩放；
     - `affine_free_rmsnorm`：逐空间 Token 无仿射 RMSNorm（显式消融项）。
3. **独立 3D 解码器（零跨层 Skip 红线）**：
   - 仅接收 `(z_clean, geometry)` 两个参数（**严格禁止任何来自 Encoder 中间特征图的 U-Net / skip connection**，以保证 Stage 2 仅凭潜变量 `z` 与已知几何即可独立解码）。
   - 3 级 **C2S（Channel-to-Space）残差上采样块** + 各尺度 2 通道 `geometry [B, 2, Z, Y, X]` 多尺度注入，输出解码特征供物理头消费。

### 2.3 `src/flow3d/models/spatial_transformer.py`（`Spatial Transformer ×1`）
1. 接收 Encoder 输出 `[B, C=256, Z_l, Y_l, X_l]`（标准网格 $8 \times 12 \times 24 = 2,304$ 个空间 Token），按 `(Z, Y, X)` 顺序展平为 `[B, N, 256]`。
2. 必须实现 **3D 空间位置编码**（3D SinCos 位置编码或 3D RoPE，支持任意 `(Z_l, Y_l, X_l)` 以便 CPU 小网格单测与 `8×12×24` 全尺寸无缝运行）。
3. 默认配置：`depth = 1`（**强制校验 `depth >= 1`，严禁 `depth = 0` 空壳绕过**）、`num_heads = 8`、`ffn_dim = 1024`、`PreLN`（LayerNorm -> MHA -> Residual；LayerNorm -> FFN/GELU-SiLU -> Residual），优先利用 PyTorch `F.scaled_dot_product_attention` 以降低 `2,304` Token 显存占用。
4. Stage 1 仅做**空间**注意力，严禁在此模块内引入跨时间维度的注意力。

### 2.4 `src/flow3d/models/heads.py`（Head-A / Head-B 双物理头 + `PhysicalReadout3D`）
严格对齐设计文档第 B、C.2、D 节及 `src/flow3d/physics/operators.py` 的 `(Z, Y, X)` 轴序约定（$u$ 沿 $X$/axis `-1`，$v$ 沿 $Y$/axis `-2`，$w$ 沿 $Z$/axis `-3`，`spacing = (dz, dy, dx)`）：
1. **PyTorch 相容离散算子（与 R1 `flow3d.physics.operators` 数值一致）**：
   - 实现可微分的 PyTorch 版 `centered_difference_3d_torch`、`div_h_torch`、`curl_h_torch`、`build_full_3d_stencil_mask_torch` 与 `valid_div_of_curl_mask_torch`，保证与 R1 NumPy 实现误差 $\le 10^{-5}$。
   - **数值精度红线**：由于两级中心差分在小网格步长（如 C04 物理网格 $h=0.01875\,\text{m}$，$1/(2h)^2 \approx 711$）下会放大低精度舍入误差，所有有限差分算子在 AMP/BF16 下必须自动升精度至至少 `float32` 计算（不得直接在 `bf16/fp16` 下求差分），且支持 `float64` 精确核验。
2. **Head-A：`DirectProjectionHead3D`（直接四场 + 可微离散投影 / 软散度约束）**：
   - 从解码器特征直接预测归一化四场 `[ũ, ṽ, w̃, p̃]`；
   - 提供可配置的离散散度处理模式：
     - `projection`：基于可微有限步 Jacobi / FFT / 泊松势修正步（或局部无散投影步）抑制有效流体内部散度；
     - `soft_penalty`：直接输出四场并返回有效内部域上的 `div_loss` 供训练加权。
   - 返回结构化输出字典（含 `norm_field`、`aux_losses` 字典及诊断量）。
3. **Head-B：`RegularizedCurlHead3D`（均值/势流分离 + 库仑规范正则的相容离散 `Curl` 头）**：
   - 针对圆柱多连通开放边界域（$b_1 = 1$）上的 Hodge–Helmholtz 分解 $\mathbf{u} = \nabla \times \mathbf{A}^* + \nabla \phi_{\text{harm}} + \mathbf{u}_{\text{mean}}$：
     - 预测矢量势 $\mathbf{A}^* = [A_x^*, A_y^*, A_z^*]$，经相容离散 `curl_h_torch` 生成无散旋度分量 $\tilde{\mathbf{u}}_{\text{curl}}$；
     - 并行预测低频均值/势流标量 $\phi$（梯度 $\nabla_h \phi$ 与空间全局/低频基流偏置），叠加构成完整速度场 $\tilde{\mathbf{u}}$（可开关纯旋度消融模式 `use_potential_branch: bool` 以便在解析无散场上直接验证 `div_h(curl_h(A*)) == 0`）；
     - 独立分支预测归一化压力 $\tilde{p}$；
     - 计算并返回库仑规范正则项 `coulomb_gauge_loss`（有效内部域上 $\|\operatorname{div}_h(\mathbf{A}^*)\|^2$ 均值），消除规范冗余 $\mathbf{A}^* \mapsto \mathbf{A}^* + \nabla \chi$。
4. **`PhysicalReadout3D`**：
   - 接收归一化四场 `norm_field [B, (T,) 4, Z, Y, X]`、每帧统计量 `b [B, (T,) 6]`（`[mu_u, mu_v, mu_w, mu_p, sigma_uvw, sigma_p]`，遵循 `normalizer.flow3d.v1`）与已知 `geometry [B, 2, Z, Y, X]`（取 `fluid_mask = geometry[:, 0:1] > 0.5`）。
   - 执行物理逆归一化：$\hat{u} = \tilde{u}\sigma_{uvw} + \mu_u$、$\hat{v} = \tilde{v}\sigma_{uvw} + \mu_v$、$\hat{w} = \tilde{w}\sigma_{uvw} + \mu_w$、$\hat{p} = \tilde{p}\sigma_p + \mu_p$，并在固体域执行硬掩码清零（`hard_mask_output=True` 时）。
   - 明确约束：固体域清零仅用于防止非流体区噪声污染，**绝不宣称其等价于物理无滑移壁面检验**（真实近壁误差须用 R1 `flow3d.physics.boundary.first_fluid_shell_mask` 评估）。

### 2.5 `src/flow3d/models/probe.py`（可选消融 `LatentPhysicsProbe`）
- 从 `z_clean [B, 256, Z_l, Y_l, X_l]` 经全局池化 + 轻量 MLP 预测无量纲拟能（enstrophy）标量与粗粒度 3D 能谱能量分箱（例如 8 个波数带），在 `stage1.yaml` 中通过 `probe.enabled: false` 默认关闭、随时可开。

### 2.6 `configs/flow3d/stage1.yaml`
- 完整记录 R2 Stage 1 模型与网格配置（`grid.spatial_shape_zyx: [64, 96, 192]`、`grid.spacing_zyx` 默认无量纲 `[0.0625, 0.0625, 0.0625]` 且前向/解码接口支持按 batch 覆盖为物理步长 `[0.01875, 0.01875, 0.01875]`、`in_channels: 6`、`latent_channels: 256`、`downsample_factor: 8`、`base_channels`、`latent_norm`、`spatial_transformer` 参数、`head.type: head_a | head_b` 及两头子参数、`physical_readout`、`probe`）。
- **兼容性要求**：必须同时可被 `pyyaml` 与无依赖的 `flow3d.data.schema.load_yaml_or_json_config` 正确解析（使用嵌套映射和行内列表 `[...]`，避免块式 `- ` 列表）。

### 2.7 R2 单元测试与全尺寸显存/吞吐基准
1. **`tests/flow3d/test_stage1_models.py`**：
   - **前后向形状与梯度流测试**：覆盖 Head-A（`projection` 与 `soft_penalty`）、Head-B（含/不含势流分支）、全部 3 种 `latent_norm` 模式、`LatentPhysicsProbe` 开启/关闭，验证 `[B, 6, Z, Y, X] -> z_clean [B, 256, Z/8, Y/8, X/8] -> [B, 4, Z, Y, X]` 形状精确且所有可训练参数梯度有限非零。
   - **零跨层 Skip 独立解码验证**：直接调用 `model.decode(z_clean, geometry)`（不经过 encoder，不传入任何 encoder 中间态），验证解码输出与 `forward` 解码结果严格一致，且更换 `geometry`（2 通道 `[B, 2, Z, Y, X]`）不会崩溃并正确反映掩码变化。
   - **`Spatial Transformer ×1` 真实生效验证**：断言 `depth >= 1`（传入 `depth=0` 必须抛出 `ValueError` / `PhysicsContractError`）；验证注意力层与 FFN 层权重梯度均非零，且修改其参数确实改变 `z_clean`。
   - **物理算子相容性与 Head-A/B 解析无散场对照验证**：
     - 使用 `src/flow3d/data/fixtures.py` 生成解析无散场，验证 PyTorch 版 `div_h_torch / curl_h_torch` 与 R1 NumPy 版 `div_h / curl_h` 数值一致（max abs diff $\le 10^{-5}$）；
     - 验证 Head-B 纯旋度分支输出在 `valid_div_of_curl_mask` 上满足离散无散恒等式：`float64` 下 $\|\operatorname{div}_h(\tilde{\mathbf{u}}_{\text{curl}})\|_\infty \le 10^{-10}$，且在 `float32` 下满足步长归一化相对残差 $\|\operatorname{div}_h(\tilde{\mathbf{u}}_{\text{curl}})\|_\infty / (\max(\|\tilde{\mathbf{u}}_{\text{curl}}\|_\infty, 10^{-8}) / \min(h)) \le 10^{-4}$（或在单位/无量纲步长 `h >= 0.0625` 下绝对残差 $\le 10^{-4}$；避免在物理小步长 `h=0.01875 m` 的 FP32 下因 $1/(2h)^2 \approx 711$ 舍入放大造成误报），同时 `coulomb_gauge_loss` 有限可导；
     - 验证 Head-A 在启用离散散度投影后有效流体内部散度 RMS 相对投影前显著下降。
   - **`PhysicalReadout3D` 契约往返测试**：与 R1 `Flow3DNormalizer` 联测，验证标准归一化后的真实/合成场经 `PhysicalReadout3D` 还原后在有效流体域内误差 $\le 10^{-5}$。
   - **配置文件解析测试**：验证 `configs/flow3d/stage1.yaml` 可通过 `load_yaml_or_json_config` 加载并直接实例化 Head-A 与 Head-B 模型。
2. **`scripts/bench_stage1_flow3d.py`（轻量全尺寸显存/吞吐基准脚本）**：
   - 在 `64×96×192`（`B=1`，FP32 与 BF16/AMP 可选）下分别测试 Head-A 与 Head-B 的前向+反向显存峰值（`torch.cuda.max_memory_allocated`）与单步耗时（ms/frame），输出 JSON 摘要供 G1 结构验收。

---

## 3. 必须执行的验证命令

```bash
# 1. 既有 R0 + R1 全部回归测试（必须 0 failed）
python -m pytest tests/test_config.py tests/test_normalization.py tests/test_geometry_physics.py tests/test_npz_dataset.py tests/test_manifest.py tests/test_checkpoint.py -q
python -m pytest tests/flow3d/test_contracts.py tests/flow3d/test_normalization.py tests/flow3d/test_axes_and_geometry.py tests/flow3d/test_physics_operators.py -q

# 2. 本轮 R2 新增模型与物理头测试（云端有 PyTorch 时必须 0 failed, 0 skipped）
python -m pytest tests/flow3d/test_stage1_models.py -q

# 3. 全量测试套件
python -m pytest tests -q

# 4. 全尺寸 64x96x192 显存与吞吐基准（云端 GPU 节点运行）
python scripts/bench_stage1_flow3d.py --config configs/flow3d/stage1.yaml --device cuda --dtype bf16
```

---

## 4. 验收标准与交付证据

1. 全量 `python -m pytest tests -q` 真实运行 100% 通过（云端装有 PyTorch 时 `tests/flow3d/` 为 **0 failed, 0 skipped**）。
2. 架构硬约束全部通过自动化单测证明：
   - 编码器-解码器零跨层 skip；
   - `Spatial Transformer ×1` 真实参与前/反向传播（拒绝 `depth=0`）；
   - 2 通道 `geometry [B, 2, Z, Y, X]` 多尺度注入与解码无崩溃；
   - Head-A 与 Head-B 均在解析无散场上通过离散物理算子一致性检验；
   - `PhysicalReadout3D` 与 `normalizer.flow3d.v1` 往返一致。
3. 提供云端 GPU 在 `64×96×192` 分辨率下 Head-A 与 Head-B 的实测参数量、峰值显存（GiB）与前/反向吞吐（ms/step），不得编造估算数字。
4. 交付汇报需包含：`git status`、新增/修改文件清单、关键模块接口签名、每条测试与基准命令的真实输出、未解决问题（P0–P3）及下一轮（R3 · G1/G2 Stage 1 训练闭环与冻结缓存导出）的最小入口建议。
