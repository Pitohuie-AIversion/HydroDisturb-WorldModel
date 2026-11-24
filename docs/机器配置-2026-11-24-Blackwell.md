# AutoDL 单卡训练机配置

采集于 2026 年 11 月 24 日 11:12（北京时间），实例为 `autodl-container-eec240973f-da9e8384`。

| 资源 | 当前容器可用规格 |
| --- | --- |
| CPU | Intel Xeon Platinum 8470Q，CPU 时间配额相当于 22 个逻辑 CPU |
| GPU | 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition，设备 `cuda:0`，计算能力 12.0 |
| 显存 | 驱动报告总量 97,887 MiB（95.59 GiB），PyTorch 可见容量 94.97 GiB |
| RAM | cgroup v2 硬上限 110 GiB，交换空间上限 0 |
| 共享内存 | `/dev/shm` 容量 55 GiB |
| 数据盘 | `/root/autodl-tmp` 文件系统容量 550 GiB |
| 软件环境 | Ubuntu 22.04.5 LTS，Python 3.12.3，PyTorch 2.8.0+cu128（CUDA 12.8），NVIDIA 驱动 580.82.09 |

宿主机有 2 颗 CPU，共 104 个物理核心、208 个逻辑 CPU，物理内存约 1,007.5 GiB。容器的 CPU 亲和范围为 `0-207`，`cpu.max` 为 `2200000 100000`，即每 100 毫秒可累计使用 2,200 毫秒 CPU 时间。规划训练资源时应采用上表的容器配额。

原始采样记录了采集时的内存占用、GPU 显存和探测来源。[查看硬件快照](../artifacts/server_capacity/blackwell-20261124/hardware.json)

本机 Flow3D Stage 1 的默认训练配置使用 BF16，批量 24，16 个 CPU 线程，三条轨迹预载入内存。实测吞吐、其他批量和启动方法见[本机调优记录](../artifacts/server_capacity/blackwell-20261124/report.md)。
