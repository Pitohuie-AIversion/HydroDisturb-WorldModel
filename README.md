# HydroDisturb-WorldModel (三维流场时空世界模型)

本项目是服务于 **[HydroDisturb-Bench](../HydroDisturb-Bench)** 的三维流场时空世界模型（Spatiotemporal World Model）深度学习架构与表示学习套件，配套交互式可视化前端仓库见 **[HydroDisturb-WorldModel-website](https://github.com/Pitohuie-AIversion/HydroDisturb-WorldModel-website)**。

整体遵循两阶段设计（Two-Stage Architecture，路线 A / FlowWorld-3D V1.3）：
- **Stage 1（三维流场表示学习与潜空间压缩）**：先把 `(u, v, w, p)` 与几何 `(fluid_mask, SDF/D)` 转换为带版本契约的规范化物理张量，再通过 3D DC-AE（Deep Compression Autoencoder，`f8 / C=256`）+ `Spatial Transformer ×1` 压缩为低维连续冻结潜变量 $z_{\text{clean}}$，结合相容流体物理算子（`div_h`、`curl_h`、有效导数掩码）与真实近壁流体层诊断进行独立验收。
- **Stage 2（潜空间时空动力学预测与世界模型）**：在 Stage 1 冻结潜变量与 SHA-256 契约哈希上，构建扰动与物理条件感知的前向时空预测模型（Mode-FM 含噪桥流匹配 / Mode-Res 一步残差 + 纯历史 `b_hat` 头），直接对接 HydroDisturb-Bench 的 C04 高雷诺数圆柱绕流与多雷诺数基准。

当前仓库已固定的核心协议与工程边界：

- 速度采用联合方差归一化（`ddof=0`，不除以 3），统计量**仅在有效二值流体掩码 `fluid_mask > 0.5` 内等权计算**（不包含网格单元体积空间权重）；前置校验有效流体域 `isfinite`，并在索引外安全清洗固体区 `NaN`，杜绝 `NaN * 0` 污染；
- 固体区域的有限差分使用 valid derivative mask（以及级联二次腐蚀掩码 `valid_div_of_curl_mask`），避免跨固体求导；
- 圆柱近壁评估显式提取第一层真实流体体素 `first_fluid_shell_mask`，拒绝把固体区清零冒充物理无滑移；
- 几何通道 `[fluid_mask, SDF/D]` 通过多尺度注入进入编码器/解码器，解码输出支持固体域硬 mask；
- 导出契约绑定归一化协议、潜变量布局和检查点 SHA-256，供 Stage 2 校验；
- 检查点保存采用临时文件 + `fsync` + `os.replace`，同时保存 Python/NumPy/Torch RNG、优化器、scheduler 和 AMP scaler（PyTorch 可选）。

## 当前项目进度

| 阶段 | 状态 | 说明 |
| :--- | :--- | :--- |
| **R1 · G0（三维数据契约与物理算子底座）** | **已完成（`f5dc000`）** | 已完成 `src/flow3d/` 核心契约、3 套本地 3D CFD 数据源适配器（含 `c04_wm_25steps_golden.h5` 合成伪数据与 `re180k` 错配自动隔离）、防泄漏切分、逐帧 6 维归一化、相容 `div_h/curl_h`、近壁第一层流体检验、`cli/inspect_data.py` 与 `tests/flow3d/` 4 套单测，并修复 `src/stage1/` 既有 P0 缺陷。 |
| **R2 · G1（Stage 1 3D DC-AE 结构与双物理头）** | **下一阶段（云端 GPU）** | 在云端 GPU 上实现 `src/flow3d/models/`（3D DC-AE S2C/C2S、`Spatial Transformer ×1`、Head-A/Head-B、`PhysicalReadout3D`、`LatentPhysicsProbe`）与 `configs/flow3d/stage1.yaml`，详见 [R2 Prompt](docs/FlowWorld3D_AI_Coding_R2_Prompt.md)。 |
| **R3–R5（Stage 1 训练 / 条件接口 / Stage 2 演化）** | 待执行 | 按 [V1.3 工程设计文档](docs/FlowWorld3D_V1_Development_Design.md) 顺次推进。 |

> 本仓库当前为已完成 G0 验收的工程与物理底座，尚未包含已训练的模型权重或最终精度结果。

## 设计与研发文档

- **[FlowWorld-3D V1.3 工程设计文档](docs/FlowWorld3D_V1_Development_Design.md)**：总体架构、数学/物理隐患修正、张量契约、目录规划与 R1–R5 验收门槛。
- **[R1 / G0 AI Coding 提示词（已完成）](docs/FlowWorld3D_AI_Coding_R1_Prompt.md)**：首轮数据底座、物理算子与 G0 探针任务规范。
- **[R2 / G1 AI Coding 提示词（云端 GPU 下一阶段）](docs/FlowWorld3D_AI_Coding_R2_Prompt.md)**：第二轮 Stage 1 3D DC-AE、Spatial Transformer 与 Head-A/B 结构任务规范。
- **[G0 实测数据探针报告](docs/g0_inspection_report.json)**：本机 3 套真实三维 CFD 数据源实测输出（含 3 条有效多雷诺数轨迹及 2 项自动隔离记录）。
- **[云端 Linux 数据配置模板](configs/flow3d/data.cloud.example.yaml)**：云端节点路径示例，明确列出应上传的 3 条核验 `.h5` 轨迹与严禁上传的隔离数据清单。

## 安装与验证

### 1. 云端 GPU 完整环境（推荐用于 R2+ 开发与训练）

```bash
python -m pip install -e ".[all]"
python -m pytest tests -q
```

其中 `.[all]` 包含 `torch>=2.2`、`h5py>=3.10`、`pyyaml>=6.0` 与 `pytest>=8`（也可按需单独安装 `.[torch]`、`.[data]`、`.[dev]`）。

### 2. 最小纯 NumPy 环境（验证 R0 + R1 契约与物理算子）

基础协议与 G0 测试只需要 NumPy + Pytest，无需 PyTorch：

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m src.flow3d.cli.inspect_data --help
```

运行云端或本地 G0 数据探针（云端请先复制 `configs/flow3d/data.cloud.example.yaml` 为 `configs/flow3d/data.cloud.yaml` 并调整路径）：

```bash
python -m src.flow3d.cli.inspect_data --config configs/flow3d/data.yaml
```

运行存量 `src/stage1/` 3D DC-AE 烟雾脚本（需要 `.[torch]` 或 `.[all]`）：

```bash
python scripts/smoke_stage1.py
```

没有 PyTorch 时，模型模块仍可被导入，但实例化会给出明确的安装提示；规范化、几何、G0 数据探针与物理算子均可独立运行。

## 目录结构

```text
HydroDisturb-WorldModel/
  docs/
    FlowWorld3D_V1_Development_Design.md  # V1.3（路线 A）完整工程设计文档
    FlowWorld3D_AI_Coding_R1_Prompt.md    # R1 / G0 任务提示词（已完成 f5dc000）
    FlowWorld3D_AI_Coding_R2_Prompt.md    # R2 / G1 Stage 1 结构提示词（云端 GPU）
    g0_inspection_report.json             # 本机 3 套真实 3D CFD 数据 G0 实测报告
  configs/
    stage1.yaml                           # 存量 Stage 1 原型配置
    flow3d/
      data.yaml                           # 本地 3 套 3D CFD 数据源与物理常数配置
      data.cloud.example.yaml             # 云端 Linux 数据路径模板与上传/隔离白名单
  src/
    flow3d/                               # V1.3 正式三维主命名空间（R1/G0 已完成 f5dc000）
      __init__.py
      contracts.py                        # StateSpec3D, Flow3DConditions, SampleProvenance, Flow3DBatch
      data/
        schema.py                         # ZYX/YZX 轴序与通道重排、无依赖 YAML/JSON 解析
        adapters.py                       # C04 NPZ / 82 帧脱落目录 / 多 Re HDF5 适配器与伪数据隔离器
        inspect.py                        # G0 结构化探针与诊断汇总
        splits.py                         # ancestor/trajectory 分组切分与 purge-gap 时间段切分
        normalization.py                  # 逐帧有效流体域 6 维归一化（ddof=0, scale_floor）
        fixtures.py                       # 3D 解析无散场与合成测试 fixture
      physics/
        operators.py                      # (Z,Y,X) 轴对齐 centered_difference_3d, div_h, curl_h 与相容性核验
        boundary.py                       # 圆柱第一层流体壳层提取、法/切向速度与进出口通量检验
      cli/
        inspect_data.py                   # G0 命令行探针入口
    stage1/                               # 既有 V1 兼容层与成熟原子检查点底座
      config.py                           # 版本化 Stage 1 配置与契约
      pipeline.py                         # 规范化、编码、解码的组合入口
      training.py                         # 原型 Stage 1 训练辅助骨架
      data/normalization.py               # 六维背景统计与可逆归一化
      data/npz.py                         # NumPy 数据集适配器与数据清单哈希
      geometry/masks.py                   # mask/SDF 规范及导数有效区域
      models/dcae.py                      # 可选 PyTorch 3D DC-AE 原型
      physics/operators.py                # 有 mask 的有限差分、散度、涡量（NumPy）
      physics/torch_operators.py          # 有 mask 的有限差分、散度、涡量（PyTorch）
      losses/reconstruction.py            # masked MSE 与物理辅助损失
      io/checkpoint.py                    # 原子检查点（mkstemp+fsync+os.replace）与恢复契约
      io/manifest.py                      # 冻结潜变量 manifest 契约校验
  tests/
    flow3d/                               # R1 / G0 单元测试套件
      test_contracts.py
      test_normalization.py
      test_axes_and_geometry.py
      test_physics_operators.py
    test_*.py                             # 既有 6 个基础兼容测试文件
  scripts/
    smoke_stage1.py                       # 存量 Stage 1 烟雾脚本
```

## 检查点与 Stage 2 绑定

训练时使用 `CheckpointManager.save_checkpoint(...)` 保存不可变的 epoch 文件；`is_best=True` 只更新带 SHA-256 的 `*_best.json` 指针。恢复时传入同一个 `CheckpointContract`，任何 latent layout、normalizer 或数据清单不一致都会直接拒绝。

```python
from stage1.io.checkpoint import CheckpointContract, CheckpointManager

contract = CheckpointContract(
    latent_layout="B,T,C,Z,Y,X",
    latent_shape=(256, 8, 12, 24),
    normalizer_version="normalizer.flow3d.v1",  # 存量 stage1 兼容层使用 "normalizer.v1"
    dataset_manifest_sha256="<dataset-manifest-sha256>",
)
manager = CheckpointManager("checkpoints", contract=contract)
manager.save_checkpoint(epoch, global_step, model, optimizer, scheduler, scaler, is_best=True)
```

Stage 2 离线缓存应从已验证的 `z_clean` 导出，并用 `manager.create_frozen_manifest(...)` 把权重哈希、契约哈希和数据清单哈希写入同一个 manifest。
