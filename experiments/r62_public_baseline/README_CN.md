# SHV R62 / PROFILE-Q33-A0：公开基准稳定性与反向峰值归因

## 研究定位
R61-FIX1 在 Tesla T4 上的 FULL/TILED/LIGER 完成对照已通过，但 TILED 相对 Liger 的 S512 峰值只低约 0.119 MiB（0.0084%），未通过 >=5% 的显存 Gate；Liger 前向耗时异常偏高，只有 3 次计时更新。

**R62 不提出新编码技术，不声称 SHV 原创性能优势。** 它仅检查公开实现之间的设备特定计时稳定性，并观察反向与损失算子的分配热点，以决定下一轮是否有值得设计新 SHV 机制的可测空间。

## 运行设计
- GPT-2 Small，batch=4，S=512，AdamW8bit，Activation Checkpointing。
- 仅比较 TILED（R60 冻结）、LIGER（外部安装，不含第三方内核源码）。
- 每个组独立 Python/CUDA 进程。三个配对重复，交替顺序 TILED/LIGER；每次 3 步预热、7 步计时。
- 核验所有模型初始哈希、输入哈希、8bit 大张量、后期训练与 Probe 损失。
- 两个独立的带插桩 Profiler 进程。Profiler 不参与速度 Gate。`self_device_alloc_bytes` 是**分配事件流量**，不等于“当时同时存活的显存”或可立即释放的字节。

## Gate
- 数值质量相对偏差 <=1%；全部重复同环境/相同初始参数、输入和量化状态。
- 只有在 >=2/3 次重复中 TILED 相比 LIGER 额外节约 >=5% 的峰值显存且耗时倍率 <=1.10，才记作公共方案的**可重复显存信号**；依旧不构成 SHV 独创资格。
- 同时报单纯速度信号：>=2/3 次重复快 >=10%。
- 不执行模型长训练、不访问 Guard、不复跑 R1–R61 历史实验。

## 运行
在 NVIDIA Tesla T4/A100 Colab 打开独立 R62 Notebook，顺序运行 5 个单元格。自动导出：
- `SHV_R62_PROFILE_Q33_A0_RESULT_UPLOAD_TO_CHATGPT.zip`
- `SHV_R62_PROFILE_Q33_A0_RESULT_UPLOAD_TO_CHATGPT.zip.sha256`

若未提供 GPU、外部包不兼容、量化状态缺失或 Profile 失败，结果包会包含可审计状态，不会误报为成功。
