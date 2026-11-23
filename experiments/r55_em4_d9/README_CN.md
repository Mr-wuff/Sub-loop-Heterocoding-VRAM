# SHV R55 / SYS-Q26-A0 · 系统级关键分岔实验（精简 Notebook）

**问题**：R54 已表明 D9 在真实 GPT-2 梯度回放中仍比 PyTorch AdamW foreach 慢约 15 倍，单个 Triton 符号位内核无法根治。R55 优先判定 SHV 的激活压缩 EM4 是否能脱离 D9，直接与高性能 AdamW 结合，形成更有工程价值的速度—显存 Pareto 方案。

## 固定对照

- N_FOREACH：原生激活 + FP32 AdamW foreach。
- N_CKPT_FOREACH：逐 Transformer Block native activation checkpoint + 相同 AdamW。
- EM4_FOREACH：冻结 EM4 激活路径 + 相同 AdamW。
- N_D9：原生激活 + 冻结 D9 FAST 2× 分块。
- EM4_D9：冻结 EM4 + 冻结 D9 FAST 2× 分块。

初始权重、文本 token、优化器超参数、梯度裁剪完全一致。D9 和 FP32 AdamW 的数字状态定义不同，仅用于系统成本对照，不要求不同优化器间逐位一致。

**两阶段安全性**：先对五组执行八步非新鲜历史数据 Pilot；若 EM4_FOREACH 不能实现至少 2% 显存节约或慢于 native 超过 1.5×，立即停止。通过时再让三组 foreach 配置完成另外两次重复；未执行新数据/B1、长训练。

## 重要限制

- 所有质量数据均为已暴露的历史受控文本；不宣称泛化或正式长期收敛。
- CPU 环境运行自动导出 `R55_GPU_NOT_AVAILABLE` 错误包；真实 CUDA 性能需要 T4/A100。
- 本实验 CPU 内不改变 EM4 或 D9 原始数学；不用 R51/R52 Triton 实验分支。
- CUDA allocated/reserved 峰值来自**训练阶段**，与训练后的独立 Native probe 分开，避免重演 R45 的评估峰值污染。
- 同时比较 activation checkpointing，以证明价值是相对于成熟方案的额外优势，而非只和未优化 native 比较。

## 文件组织

`shv_r55/activation.py`、`shv_r55/frozen_d9.py` 是 R54 冻结源码的 SHA256 一致副本；`shv_r55/runner.py` 是新的系统比较脚本。Notebook 只负责依赖、解包并调用脚本。运行完回传：

`SHV_R55_SYS_Q26_A0_RESULT_UPLOAD_TO_CHATGPT.zip` 和对应 `.sha256`。
