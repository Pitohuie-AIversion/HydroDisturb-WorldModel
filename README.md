# HydroDisturb-WorldModel (三维流场时空世界模型)

本项目是服务于 **[HydroDisturb-Bench](../HydroDisturb-Bench)** 的三维流场时空世界模型（Spatiotemporal World Model）深度学习架构与表示学习套件，配套交互式可视化前端仓库见 **[HydroDisturb-WorldModel-website](https://github.com/Pitohuie-AIversion/HydroDisturb-WorldModel-website)**。

整体遵循两阶段设计（Two-Stage Architecture，路线 A / FlowWorld-3D V1.3）：
- **Stage 1（三维流场表示学习与潜空间压缩）**：先把 `(u, v, w, p)` 与几何 `(fluid_mask, SDF/D)` 转换为带版本契约的规范化物理张量，再通过 3D DC-AE（Deep Compression Autoencoder，`f8 / C=256`）+ `Spatial Transformer ×1` 压缩为低维连续冻结潜变量 $z_{\text{clean}}$，结合相容流体物理算子（`div_h`、`curl_h`、有效导数掩码）与真实近壁流体层诊断进行独立验收。
- **Stage 2（潜空间时空动力学预测与世界模型）**：在 Stage 1 冻结潜变量与 SHA-256 契约哈希上，构建扰动与物理条件感知的前向时空预测模型（Mode-FM 含噪桥流匹配 / Mode-Res 一步残差 + 纯历史 `b_hat` 头），直接对接 HydroDisturb-Bench 的 C04 高雷诺数圆柱绕流与多雷诺数基准。

当前仓库已固定的核心协议与工程边界：

- HDF5 入库时将速度和压力转为 m/s 和 Pa。NPZ 与 HDF5 共用 SI 场与 SI 统计量 `b`，`batch.spacing` 单位为 m，归一化场的物理头使用 `batch.spacing_over_d(i)`。
- 速度采用联合方差归一化（`ddof=0`，不除以 3），统计量**仅在有效二值流体掩码 `fluid_mask > 0.5` 内等权计算**（不包含网格单元体积空间权重）；前置校验有效流体域 `isfinite`，并在索引外安全清洗固体区 `NaN`，杜绝 `NaN * 0` 污染；
- 固体区域的有限差分使用 valid derivative mask（以及级联二次腐蚀掩码 `valid_div_of_curl_mask`），避免跨固体求导；
- 圆柱近壁评估显式提取第一层真实流体体素 `first_fluid_shell_mask`，拒绝把固体区清零冒充物理无滑移；
- 几何通道 `[fluid_mask, SDF/D]` 通过多尺度注入进入编码器/解码器，解码输出支持固体域硬 mask；
- 导出契约绑定归一化协议、潜变量布局和检查点 SHA-256，供 Stage 2 校验；
- 检查点通过临时文件和 `fsync` 落盘，独占发布后原子更新最佳指针，同时保存 Python/NumPy/Torch RNG、优化器、scheduler 和 AMP scaler（PyTorch 可选）。

## 当前项目进度

| 阶段 | 状态 | 说明 |
| :--- | :--- | :--- |
| **R1 · G0（三维数据契约与物理算子底座）** | **已完成（`f5dc000`）** | 已完成 `src/flow3d/` 核心契约、3 套本地 3D CFD 数据源适配器（含 `c04_wm_25steps_golden.h5` 合成伪数据与 `re180k` 错配自动隔离）、防泄漏切分、逐帧 6 维归一化、相容 `div_h/curl_h`、近壁第一层流体检验、`cli/inspect_data.py` 与 `tests/flow3d/` 4 套单测，并修复 `src/stage1/` 既有 P0 缺陷。 |
| **R2 · G1（Stage 1 结构与双物理头）** | 已完成结构验证 | 3D DC-AE 和两种物理头已通过单卡全尺寸 BF16 前后向测试。重构精度随 R3 训练验收。 |
| **R3 · G2（冻结表示筛选）** | 已完成十步验证 | 在两个独立验证祖先、8 个起点上用相同的轻量预测器筛选，A830 的十步速度和压力 RMSE 均低于 B475。 |
| **R4–R5（条件与时序预测）** | 确定性流程已完成十步验证 | 五类条件与四帧历史驱动 Clean/Query Mode-Res，逐步预测潜变量和六维背景统计量，解码得到 SI 流场。预留测试、Mode-FM、稳态频谱和物理干预待评。 |

按工况拆开后，十步全场收益集中在 Re=500,000。Re=40,000 的真实流场线性外推误差更低，两条验证轨迹上 H20、H45 的平均误差均以保持最后一帧更低。[分工况验证、近壁误差和独立重放](docs/FlowWorld3D_Validation_Calibration_20261124.md)

训练权重、冻结缓存和物理场预测保存在本地运行目录。下方命令可从一次有数据划分记录的 Stage 1 训练生成这些产物。

## 设计与研发文档

- **[FlowWorld-3D V1.3 工程设计文档](docs/FlowWorld3D_V1_Development_Design.md)**：总体架构、数学/物理隐患修正、张量契约、目录规划与 R1–R5 验收门槛。
- **[R1 / G0 AI Coding 提示词（已完成）](docs/FlowWorld3D_AI_Coding_R1_Prompt.md)**：首轮数据底座、物理算子与 G0 探针任务规范。
- [R2 / G1 结构任务与验收](docs/FlowWorld3D_AI_Coding_R2_Prompt.md)
- [G0 数据报告](docs/g0_inspection_report.json)保留两组 C04 NPZ 的历史探针，并更新多 Re HDF5 的逐帧核验。当前 3 条轨迹可作瞬态预测，Tier-1 稳态通过数与成对干预数均为 0。
- **[云端 Linux 数据配置模板](configs/flow3d/data.cloud.example.yaml)**：云端节点路径示例，明确列出应上传的 3 条核验 `.h5` 轨迹与严禁上传的隔离数据清单。

## 安装与验证

### 使用 uv 运行测试

在仓库根目录运行下列命令。uv 会按 `uv.lock` 同步 Python 3.12 项目环境和全部测试依赖，pytest 会自动发现 `tests/` 下的用例。

```bash
uv run --locked --python 3.12 --extra all python -m pytest -q
```

仅做数据校验和轻量测试时，使用 `uv sync --locked --python 3.12 --extra data --extra dev`。修改 `pyproject.toml` 中的依赖后运行 `uv lock` 更新锁文件，可用 `uv lock --check` 检查两者是否一致。

### 1. 云端 GPU 完整环境（推荐用于 R2+ 开发与训练）

当前 AutoDL 单卡实例的 CPU 配额、GPU 和内存见[机器配置快照（2026-11-24）](docs/机器配置-2026-11-24-Blackwell.md)。

> **云端版本护栏**：请使用 **PyTorch `2.5.1–2.8.0`**（避开 PyTorch `2.9.0/2.9.1+cu128/cu129` 的 `Conv3d` BF16 cuDNN 显存与性能回归 `pytorch#166643/#166790/#168167`）与 **`h5py>=3.14`**（兼容 `numpy>=2.0` 且修复 Linux `fork` 死锁）。若云端镜像已预装匹配驱动的 CUDA PyTorch，建议用 `--no-deps` 安装本包以免覆盖预装 `torch`：

```bash
# 若云端镜像已预装 CUDA PyTorch (2.5.1 ~ 2.8.x)：
python -m pip install --no-deps -e .
python -m pip install "h5py>=3.14" "pyyaml>=6.0" "pytest>=8" matplotlib tensorboard

# 若为全新虚拟环境：
python -m pip install -e ".[all]"

python -m pytest tests -q
```

其中 `.[all]` 包含 PyTorch、HDF5、YAML、pytest、Matplotlib 和 TensorBoard。也可按需单独安装 `.[torch]`、`.[data]`、`.[dev]` 或 `.[tensorboard]`。

### 用 TensorBoard 查看 Stage 1 训练

默认 Stage 1 配置启用 TensorBoard。在仓库根目录运行训练，在另一个终端打开面板。

```bash
uv run --locked --extra all python scripts/train_stage1_flow3d.py --save-dir checkpoints/stage1
uv run --locked --extra all tensorboard --logdir checkpoints --port 6006
```

浏览器打开 `http://localhost:6006`。`step/` 曲线按成功的优化器更新记录损失、梯度范数和学习率。`train/` 和 `validation/` 曲线按 epoch 显示，验证指标包括速度 RMSE（m/s）、压力 RMSE（Pa）、散度和近壁误差。

每次启动会在 `<save-dir>/tensorboard/` 下建立独立运行目录，完整路径也会写入 `training_summary.json`。从检查点续训时，新目录会先写入检查点保留的 epoch 历史，逐步指标从恢复后的下一次更新开始。可用 `--tensorboard-dir PATH` 指定父目录，或用 `--no-tensorboard` 关闭记录。

### 从已有 Stage 1 权重跑到多步流场预测

使用一次已完成训练的 Stage 1 目录，该目录应含最佳权重指针及记录了数据划分的 `training_summary.json`。`--dataset-dir` 指向这次训练使用的 HDF5 数据与清单。在仓库根目录执行，可将 `--device cuda:0` 改成 `--device cpu`。

```bash
uv run --locked --extra all python -m flow3d.cli.forecast run \
  --checkpoint-dir checkpoints/stage1 --dataset-dir /data/world_model \
  --out-dir artifacts/flow3d_demo --device cuda:0
```

默认使用 4 帧历史、2 步预测，训练 Stage 2 一轮。它只从训练段取窗口，滚动时使用自己的上一步预测，并沿用最后一帧观测的背景统计量 `b`。评估会在同一批测试窗口上比较 Mode-Res、沿用最后真实物理场的 Persistence 和外推最后一个潜变量增量的 Delta-z。每条测试段至少需要 6 帧。

输出目录包含冻结缓存 `cache/cache_manifest.json`、时序模型 `forecaster.pt`、逐时距评分 `rollout_report.json` 和物理场样例 `forecast_example.npz`。缓存可用 `run --cache-manifest PATH` 复用。单独预测只需缓存和时序模型，默认从所选轨迹段的最后一帧开始。

```bash
uv run --locked --extra all python -m flow3d.cli.forecast predict \
  --cache-manifest artifacts/flow3d_demo/cache/cache_manifest.json \
  --model artifacts/flow3d_demo/forecaster.pt --horizon 2 --device cuda:0 \
  --output artifacts/flow3d_demo/future.npz
```

NPZ 的 `fields` 数组采用 `[T,C,Z,Y,X]` 布局，通道依次为 `u,v,w,p`，单位为 m/s、m/s、m/s、Pa。文件还保存流体掩码、预测起点、物理秒和无量纲对流时间。用 `--trajectory` 和 `--origin-step` 可选择缓存中的其他预测起点。

#### G2 十步验证与条件预测

以下命令在仓库根目录的 Bash 中运行。示例复用已登记的 G0 审核和下游划分，`g0_audit.json` 与 `split_manifest.json` 应在同一目录。Stage 1 原始数据目录用于核对训练来源，新的 HDF5 目录提供下游轨迹。每个候选使用独立的新缓存目录，仅编码训练与验证祖先。

```bash
py=.venv/bin/python
stage1=/root/autodl-tmp/model/g1_ab_plateau_20261123
data=/root/autodl-tmp/datasets/world_model
split=/root/autodl-tmp/model/g2_tenstep_v1/data/split_manifest.json
out=artifacts/g2_quickstart

for head in head_a head_b; do
  case "$head" in
    head_a) digest=5e1e2b36988ef0d30795a6bbf336faab3906247be456196cf9f80d48bda21429 ;;
    head_b) digest=493e1239be8948c17330df8f2955b360f32838f596aab45b4a64e7a112518366 ;;
  esac
  "$py" -m flow3d.cli.export_g2_cache \
    --checkpoint-dir "$stage1/validation_extension_2300/$head" --stage1-data-dir "$stage1/data" \
    --dataset-dir "$data" --split-manifest "$split" --expected-checkpoint-sha256 "$digest" \
    --out-dir "$out/$head/cache" --device cuda:0 || break
  "$py" -m flow3d.cli.forecast run --model channelwise \
    --cache-manifest "$out/$head/cache/cache_manifest.json" --dataset-dir "$data" \
    --out-dir "$out/$head/probe" --eval-split val --history 4 --train-horizon 2 --horizon 10 \
    --origin-stride 10 --epochs 12 --batch-size 4 --lr 0.005 --seed 42 --device cuda:0 --no-example || break
done
```

两个 `probe/rollout_report.json` 记录相同验证起点上逐时距的速度与压力 RMSE，以及 Persistence、Delta-z 的结果。按预先登记的第十步规则选头后，将下例的 `selected_head` 设为对应目录名。Clean/Query Mode-Res 使用缓存中的工况和历史帧，同时预测潜变量与背景统计量 `b`。

```bash
selected_head=head_a  # 按本次验证选择 head_a 或 head_b
cache="$out/$selected_head/cache/cache_manifest.json"
"$py" -m flow3d.cli.forecast run --model clean-query \
  --cache-manifest "$cache" --dataset-dir "$data" --out-dir "$out/clean_query" \
  --eval-split val --history 4 --train-horizon 2 --horizon 10 --origin-stride 10 \
  --epochs 6 --batch-size 2 --lr 0.0002 --seed 42 --device cuda:0

"$py" -m flow3d.cli.forecast predict --cache-manifest "$cache" \
  --model "$out/clean_query/forecaster.pt" --split val --trajectory re40k_wm_50steps \
  --origin-step 855 --horizon 10 --device cuda:0 --output "$out/clean_query/re40k_from_855.npz"
```

最后一条命令只需冻结缓存及其配套的时序权重，从第 855 步的历史窗口滚动生成第 856 至 865 步的四通道 SI 流场。

**条件数据的兼容性。** 推理需要已知、有效的物理步长（秒）和对流步长（CTU）。后者可直接给定，也可由 `dt_phys_s × |u0_m_s| / diameter_m` 计算。旧 JSON 若在缺少参考速度时曾把秒数写进 `dt_ctu`，请从源时间轴补正，无法核实时将其设为 `null`。条件编码器可表示缺失步长，滚动预测会要求补齐。

带动作的缓存保存其外源声明 `extra_metadata.exogenous = true`，只在动作已知且发布时间早于或等于预测起点时使用。旧缓存若缺少这项声明，可在 `predict` 命令中加 `--dataset-dir "$data"`。读取器按冻结哈希核对原始 G0 资料、动作值、单位和时间轴，在内存中恢复声明，已有时序模型仍可使用原缓存。直接传入的条件字典须显式声明外源动作。

### 2. 最小纯 NumPy 环境（验证 R0 + R1 契约与物理算子）

基础契约和差分算子可在 NumPy 环境中验证。HDF5 完整准入需要 `.[data]` 中的 h5py。

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/flow3d/test_contracts.py tests/flow3d/test_normalization.py tests/flow3d/test_axes_and_geometry.py tests/flow3d/test_physics_operators.py -q
python -m src.flow3d.cli.inspect_data --help
```

运行云端或本地 G0 数据探针（云端请先复制 `configs/flow3d/data.cloud.example.yaml` 为 `configs/flow3d/data.cloud.yaml` 并调整路径）：

```bash
python -m src.flow3d.cli.inspect_data --config configs/flow3d/data.yaml --strict
# 只检查已挂载的多 Re 数据集时，将路径替换为本机目录
python -m src.flow3d.cli.inspect_data --path /data/hydrodisturb/03_world_model_dataset --strict
```

认证 C04 四帧基准时，在配置中填写从受信发布记录取得的 `c04_benchmark_sha256`，或为单文件检查传入 `--expected-sha256`。缺少该值的有效文件标记为 `UNVERIFIED`。

Stage 1 的全尺寸设备、精度和梯度检查可在指定 GPU 上运行。

```bash
python scripts/smoke_flow3d_stage1.py --device cuda:0 --amp --shape 1 1 64 96 192
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
        boundary.py                       # 圆柱第一层流体壳层提取、法/切向速度与采样区域六面向外流量
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

训练时使用 `CheckpointManager.save_checkpoint(...)` 按 epoch 和 step 保存文件。同一步重复保存时，已有文件保留，调用方使用保存方法返回的路径绑定缓存。`is_best=True` 会更新带 SHA-256 的 `*_best.json` 指针。恢复时传入同一个 `CheckpointContract`，任何 latent layout、normalizer 或数据清单不一致都会直接拒绝。

```python
from stage1.io.checkpoint import CheckpointContract, CheckpointManager, sha256_file

contract = CheckpointContract(
    latent_layout="B,T,C,Z,Y,X",
    latent_shape=(256, 8, 12, 24),
    normalizer_version="normalizer.flow3d.si.v2",
    dataset_manifest_sha256=sha256_file("dataset/manifest.json"),
    extra={"latent_contract_version": "latent.flow3d.v2", "latent_norm": "group_norm",
           "spatial_output_norm": "none"},
)
manager = CheckpointManager("checkpoints", contract=contract)
weights = manager.save_checkpoint(epoch, global_step, model, optimizer, scheduler, scaler, is_best=True)
```

导出器应在 `model.eval()` 下生成 `z_clean`，将索引和分片写入单独的版本目录。JSON 索引保存实际编码 checkpoint 的 `checkpoint_sha256`、该 checkpoint 内的 `contract_sha256` 和有序的 `shards` 相对路径列表。索引可包含样本与轨迹映射，整个文件都计入哈希。分片写完后发布 manifest，随后保持该目录只读。

```python
import json
from pathlib import Path

cache_dir = Path("exports/run_001").resolve()
manifest = cache_dir / "cache_manifest.json"
# shard_names 由实际加载 weights 的导出器按分片写入顺序记录
index = {
    "checkpoint_sha256": sha256_file(weights),
    "contract_sha256": manager.verify_checkpoint(weights, expected_contract=contract)["contract_sha256"],
    "shards": shard_names,
}
(cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
manager.create_frozen_manifest(weights, cache_index="index.json", output_path=manifest)

# Stage 2 使用来自本次数据集与模型配置的预期契约
verified = manager.verify_frozen_manifest(manifest, expected_contract=contract)
index_path = verified["cache_index_path"]
shard_paths = verified["shard_paths"]  # 与索引中的顺序一致
```

v1 manifest 只有 checkpoint 元数据。旧缓存若包含按星号量计算的 `b`、步号生成的时间或默认祖先 ID，应从核验后的源文件重新导出。只有使用相同单位、时间、来源和 checkpoint，并持有完整导出记录的缓存，才可直接生成索引并发布 v2 manifest。
