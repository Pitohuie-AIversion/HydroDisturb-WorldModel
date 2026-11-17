# HydroDisturb-WorldModel (三维流场时空世界模型)

本项目是服务于 **[HydroDisturb-Bench](../HydroDisturb-Bench)** 的三维流场时空世界模型（Spatiotemporal World Model）深度学习架构与表示学习套件。

整体遵循两阶段设计（Two-Stage Architecture）：
- **Stage 1（三维流场表示学习与潜空间压缩）**：先把 `(u, v, w, p, mask, sdf)` 转换为带版本契约的规范化物理张量，再通过 3D DC-AE（Discrete Continuous Autoencoder）压缩为低维连续冻结潜变量 $z$，并结合流体物理算子（散度、涡量、导数有效掩码）与重构损失进行独立验收。
- **Stage 2（潜空间时空动力学预测与世界模型）**：在 Stage 1 冻结潜变量与契约哈希上，构建扰动感知的前向预测模型（如时空 DiT / Flow Matching），直接对接 HydroDisturb-Bench 的 C04 高雷诺数扰动基准。

当前仓库刻意把协议和工程边界先固定下来：

- 速度采用联合方差归一化，空间权重和 mask 显式参与统计；
- 固体区域的导数使用 valid derivative mask，避免跨固体求导；
- 几何通道通过多尺度注入进入编码器/解码器，解码输出支持硬 mask；
- 导出契约绑定归一化协议、潜变量布局和检查点 SHA-256，供 Stage 2 校验；
- 检查点保存采用临时文件 + `os.replace`，同时保存 Python/NumPy/Torch RNG、优化器、scheduler 和 AMP scaler（Torch 可选）。

## 安装与验证

基础协议测试只需要 NumPy：

```powershell
python -m pip install -e .
pytest -q
```

要运行 3D DC-AE 前向，需要安装 PyTorch：

```powershell
python -m pip install -e .[torch]
python scripts/smoke_stage1.py
```

没有 PyTorch 时，模型模块仍可被导入，但实例化会给出明确的安装提示；规范化、几何和物理算子可以独立运行。

## 目录

```text
src/stage1/
  config.py                 # 版本化 Stage 1 配置与契约
  pipeline.py               # 规范化、编码、解码的组合入口
  data/normalization.py     # 六维背景统计与可逆归一化
  data/npz.py               # NumPy 数据集适配器与数据清单哈希
  geometry/masks.py         # mask/SDF 规范及导数有效区域
  models/dcae.py            # 可选 PyTorch 3D DC-AE
  physics/operators.py      # 有 mask 的有限差分、散度、涡量
  losses/reconstruction.py  # masked MSE 与物理辅助损失
  io/checkpoint.py          # 原子检查点与恢复契约
```

这是第一阶段的工程基础，不会伪装成已完成训练的模型权重或已验证的精度结果。

## 检查点与 Stage 2 绑定

训练时使用 `CheckpointManager.save_checkpoint(...)` 保存不可变的 epoch 文件；`is_best=True` 只更新带 SHA-256 的 `*_best.json` 指针。恢复时传入同一个 `CheckpointContract`，任何 latent layout、normalizer 或数据清单不一致都会直接拒绝。

```python
from stage1.io.checkpoint import CheckpointContract, CheckpointManager

contract = CheckpointContract(
    latent_layout="B,T,C,Z,Y,X",
    latent_shape=(256, 8, 12, 24),
    normalizer_version="normalizer.v1",
    dataset_manifest_sha256="<dataset-manifest-sha256>",
)
manager = CheckpointManager("checkpoints", contract=contract)
manager.save_checkpoint(epoch, global_step, model, optimizer, scheduler, scaler, is_best=True)
```

Stage 2 离线缓存应从已验证的 `z_clean` 导出，并用 `create_frozen_manifest(...)` 把权重哈希、契约哈希和数据清单哈希写入同一个 manifest。
