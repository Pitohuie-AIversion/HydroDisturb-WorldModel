# FlowWorld-3D 确定性预测的验证复核

2026-11-24。冻结 A830 后，模型能够从四帧三维 CFD 历史连续生成十帧速度和压力。原登记的十步误差可以从原始 HDF5 独立复现。预测收益集中在 Re=500,000 的全流体域；Re=40,000、第一层近壁速度和更长时距暴露出不同的问题。正式 G3 测试仍待评。[原实验与权重说明](/root/autodl-tmp/model/g2_tenstep_v1/summary.md)

## 十步预测在哪些工况有收益

每格为第十步的速度向量 RMSE（m/s）与压力 RMSE（Pa），数值越小越好。每条轨迹取四个固定起点，在有效流体格点算误差，先平均起点，再等权平均两条轨迹。时序预测器训练祖先的 Re 分别为 300,000、3,000,000，因此 40,000 在训练范围外，500,000 在范围内。每个验证 Re 只有一条轨迹。[登记方案](/root/autodl-tmp/model/g2_tenstep_v1/clean_query_protocol.json) · [数据划分](/root/autodl-tmp/model/g2_tenstep_v1/data/split_manifest.json)

| 方法 | Re=40,000 | Re=500,000 | 两轨迹均值 |
|---|---:|---:|---:|
| A830 + Clean/Query | 0.01675 / 0.349 | **0.18735 / 43.09** | 0.10205 / **21.72** |
| A830 + 逐通道轻量预测器 | 0.01154 / 0.371 | 0.19040 / 54.18 | **0.10097** / 27.28 |
| A830 潜变量线性外推（Delta-z） | 0.00787 / 0.240 | 0.22079 / 58.42 | 0.11433 / 29.33 |
| 最后真实流场保持不变 | 0.00994 / 0.318 | 0.23441 / 61.48 | 0.12217 / 30.90 |
| 最后两帧真实流场线性外推 | **0.00275 / 0.094** | 0.26188 / 62.98 | 0.13232 / 31.54 |

对照原登记的 Delta-z，Clean/Query 的十步平均速度和压力误差分别下降 10.74% 和 25.95%。新增的物理空间外推直接使用起报时已知的最后两帧，在相同等步长片段逐格计算。在 Re=40,000 上，Clean/Query 的误差分别是这个简单基线的 6.08 倍和 3.72 倍。Re=500,000 的绝对误差量级更大，对两轨迹 SI 平均值的影响也更大。[逐起点结果](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/physical_baselines.json) · [分工况逐时距曲线](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/validation_by_re.png)

作为补充，对每个目标帧用有效流体区的波动幅度缩放误差。速度除以三个分量空间方差之和的平方根，压力除以压力的空间标准差。所得十步均值分别为 Clean/Query **0.418 / 0.166**，Delta-z **0.368 / 0.168**，真实流场线性外推 **0.368 / 0.141**。后者两项都优于 Clean/Query。原登记的 SI 指标回答误差有多大，补充指标回答误差相当于该工况自身波动的多少。这个定义借鉴 The Well 的方差缩放，并针对联合速度和流体掩码作了调整。[The Well 原论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/4f9a5acd91ac76569f2fe291b1f4772b-Paper-Datasets_and_Benchmarks_Track.pdf) · [指标和原始值](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/physical_baselines.json)

## 长时距、近壁与表示误差

以下为看过原十步结果后固定的补充验证。H 表示起报后的输出步数。两条验证轨迹的 H10 分别覆盖 0.500 s（0.264 个对流时间）和 0.150 s（0.991 个对流时间）。H10 枚举全部可用起点；H20、H45 从首个可用起点起，每隔十帧取一次。重叠窗口共享 CFD 状态，各行的独立祖先数始终为二。[补充协议](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/sensitivity_protocol.json)

| 终点 | 实测起点数 | Clean/Query（m/s / Pa） | 保持最后一帧（m/s / Pa） | Delta-z（m/s / Pa） |
|---|---:|---:|---:|---:|
| H10 | 74 | 0.10237 / 21.82 | 0.12325 / 30.94 | 0.11531 / 29.50 |
| H20 | 6 | 0.22315 / 58.30 | 0.17312 / 44.88 | 0.24502 / 72.16 |
| H45 | 2 | 0.38767 / 177.30 | 0.06410 / 14.11 | 0.37436 / 129.98 |

扩大十步覆盖后，Clean/Query 对 Delta-z 的 SI 平均优势仍在。到 H20 和 H45，保持最后一帧的两个误差均更低。按约一个对流时间对齐，两个工况各查一个起点，Re=40,000 的 H38 仍是 Delta-z 更准（0.02713 / 0.834，对应 Clean/Query 的 0.07613 / 1.113）。Re=500,000 的 H10 仍是 Clean/Query 更准（0.18221 / 42.52，对应 Delta-z 的 0.21098 / 56.22）。这只能定位两个样例的行为，尚不足以估计跨轨迹稳定性。[各工况结果与物理时距](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/sensitivity.json)

- **近壁与散度。** 在原八个起点，取与固体面相邻的第一层流体格点算速度向量 RMSE，在六邻居均为流体的内部格点算预测与 CFD 离散散度之差的 RMS。Re=500,000 的 Clean/Query 分别为 **0.18366 m/s、0.51392 s⁻¹**，Delta-z 为 **0.10805 m/s、0.33956 s⁻¹**；Re=40,000 的 Clean/Query 为 **0.04299 m/s、0.03487 s⁻¹**，Delta-z 为 **0.04152 m/s、0.02981 s⁻¹**。在两个工况上，保持最后一帧的这两项误差均低于 Clean/Query 和 Delta-z。PDEBench 分开评估全场、边界和频域误差；这里的第一层流体和内部散度采用本项目网格定义。[PDEBench 附录 B](https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Supplemental-Datasets_and_Benchmarks.pdf) · [物理误差复算](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/replay_validation.json)
- **近壁误差从哪里出现。** 原八个起点的近壁层只有 3,072 个格点，占流体格点的 0.263%。Re=40,000 下，实际两帧物理外推、真实目标编解码、Clean/Query 的近壁速度误差依次为 **0.00080、0.03950、0.04299 m/s**。在冻结表示做单帧往返时，就已出现与完整预测相近的误差。Re=500,000 下三者为 **0.02701、0.07562、0.18366 m/s**，完整滚动的近壁误差进一步增大。同时，Clean/Query 在该工况其余流体格点上优于 Delta-z（**0.18736** 对 **0.22101 m/s**）。这解释了全场均值和近壁指标可以给出不同排序。[同一格点上的分区对照](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/shell_comparison.json) · [复算脚本](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/shell_comparison.py)
- **冻结表示。** 另将真实目标帧直接编码、用其真实背景量解码，在各 37 个 H10 终点单独测量编解码误差。Re=40,000 为 **0.00700 m/s、0.249 Pa**，同批 Delta-z 为 **0.00778 m/s、0.237 Pa**。Re=500,000 的编解码误差为 **0.01410 m/s、5.176 Pa**，远低于完整预测的 **0.18803 m/s、43.29 Pa**。低 Re 值提示应先检查表示能否保留微弱变化，高 Re 则有较大的时序预测改进空间。这个诊断看到了真实未来，只衡量编解码往返，不能作为可部署基线或预测误差下界。[编解码对照](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/sensitivity.json)
- **条件输入。** 保留真实时间步、历史和动作可见性，一组同时掩掉四项静态物性与结构化文本，另一组在两工况间交换这五项输入。两个工况的 H10 平均速度、压力误差相对原配置的最大变化小于 **0.003%**。历史和时间步仍可携带工况信息。目前合格的同起点干预对为 0，稳态轨迹为 0，因而还无法从这批数据评估真实扰动响应或稳态谱。[条件实验](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/condition_ablation.json) · [数据审计](/root/autodl-tmp/model/g2_tenstep_v1/data/g0_audit.json)

## 架构归因与可复现性

- **表示与时序。** DC-AE 提供了残差压缩架构，并讨论图像潜变量的尺度。Wiewel 等在三维流体压力上验证过压缩表示加时序预测。本项目的 A830 使用三维 f8/C256、2,304 个空间位置、GroupNorm 和数值为 1.0 的全局尺度，Head-A 训练含权重 0.01 的散度项。这些取值和本次速度、压力精度均来自本地模型。[DC-AE 附录 D](https://arxiv.org/html/2410.10733v8#A4) · [Latent-space Physics](https://arxiv.org/abs/1802.10123) · [A830 训练摘要](/root/autodl-tmp/model/g1_ab_plateau_20261123/validation_extension_2300/head_a/training_summary.json) · [工程设计](FlowWorld3D_V1_Development_Design.md)
- **动力学与验收。** 这里的 Clean/Query 用最近的潜变量增量初始化，再加上按无量纲时间步缩放的学习修正。训练损失覆盖两步，推理用自身的预测回填窗口。MeanFlow 原论文学习平均速度与瞬时速度的关系；文档已按实际目标更名。Kohl 等把尾流的预测精度、时间稳定性和谱统计分别评估，其二维实验为安排验证提供方法参考，本项目的三维结论取自上表。[MeanFlow](https://arxiv.org/abs/2505.13447) · [Kohl 等的流场预测实验](https://arxiv.org/html/2309.01745v2)
- **独立重放。** 从两份验证 HDF5 重新读取四帧历史，独立换算 SI、计算背景量并重新编码。在两套头的 16 组历史中，潜变量与背景量都和缓存逐元素一致。随后只用起报前可见数据生成未来，六种方法 × 八个起点 × 十个时距的 **480 条**记录与登记评分比较，速度、压力最大绝对差分别为 **5.56×10⁻¹⁷ m/s、3.56×10⁻¹⁴ Pa 以内**。预留测试祖先的评分帧数为 0。[重放脚本](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/replay_validation.py) · [逐条核验](/root/autodl-tmp/model/g2_tenstep_v1/internet_calibration/replay_validation.json)
- **条件契约修复。** 无法推导无量纲步长时保留缺失，预测时要求补齐；外源动作须有显式声明，旧缓存可用原始准入记录验证。修复后重放一条十步预测，导出的完整流场与其他 11 个数组逐字节一致。80 条验证记录的外源输入与时间步整体 SHA-256 在修复前后相同。全仓测试 209 项通过，2 项因本机只提供单 GPU 而跳过。[修复后重放](/root/autodl-tmp/model/g2_tenstep_v1/condition_contract_replay.json) · [回归日志](/root/autodl-tmp/model/g2_tenstep_v1/condition_contract_regression.log)

## 下一次实验的顺序

1. **先做一个低成本的表示对照。** 保留 A830 为基线，在原 Stage 1 训练集上从同一权重短程微调，对照「沿用原损失」和「增加第一层流体速度项」。固定数据、步数和种子，先用验证轨迹的真实目标编解码检查近壁改善与主体流区代价。若保留新表示，须为其重新导出缓存并训练配套的轻量时序预测器。
2. **再验证可迁移的时距。** 用新的独立 CFD 初始状态，在预先选定的对流时距报告每个工况的速度、压力和近壁误差，把沿用末帧及真实两帧外推都列入对照。当前两条轨迹的重叠起点只够做开发诊断。
3. **最后做正式测试。** 锁定一个候选、主时距和计分方法后，在预留的 Re=80,000 与 700,000 上一次性评分。Mode-FM 集合、稳态谱、同源干预分支各有独立的数据和验收条件。
