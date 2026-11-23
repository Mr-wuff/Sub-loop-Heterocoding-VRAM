# SHV R60 / LC-Q31-A0：输出消费者分块重计算机制预实验

本轮 **不使用 D9/EM4，不调整旧的冻结评价**。基准是 R59 的强组合 AdamW8bit + Transformer Activation Checkpointing。本轮只改变语言模型输出消费者的损失计算：

- FULL：一次性计算全部**有效 token**的 `[B,S-1,V]` logits（不计算最后一个无监督 token），做标准 shifted CE；
- TILED：按 token 分片计算 `F.linear + CE`，使用 `torch.utils.checkpoint(..., use_reentrant=False)` 在反向时重算单片 logits。

这是**通用的低峰值输出损失实现/系统成本研究**，不是独有 SHV 新算法，也不构成长训练合格证据。若通过，可作为之后所有 SHV 候选共同采用的更强系统基准。

### 实验门槛

先对 CPU/GPU 合成张量比较 loss 与 hidden/weight 梯度，然后在同一 T4 环境、同初始 GPT-2 权重和固定数据上测 S128 FULL、TILED。若出现完整性问题、>1% 损失偏差、峰值上涨 >5% 或耗时 >1.5×，自动停止。只有 S128 通过后才测 S512：要求对 FULL 峰值下降 ≥5%，整步耗时 ≤1.25×，训练轨迹与诊断 Probe 相对差 ≤1%。

输出所有阶段的真实 CUDA `peak allocated`、逐步 forward/backward/optimizer 时间、常驻 8-bit 优化器状态字节、原生 Checkpointing 是否生效，以及完整 ZIP / SHA-256。各组由**独立 Python CUDA 进程**执行。仅有一个固定数据顺序、各 1 预热 + 3 测量：为机制发现，不是最终资格。

**注意**：S128/512 运行指标必须与本轮 FULL 在同一执行环境比较；R59 指标只是只读溯源，不用于 R60 资格判断。CUDA OOM、缺依赖或版本接口不兼容均会导出 HOLD/ERROR 证据包。

### Colab 运行

上传 `SHV_R60_LC_Q31_A0_LEAN_Colab.ipynb`，选择 Tesla T4/A100，顺序运行。脚本会根据需要安装 `transformers`、`bitsandbytes`，然后解包内嵌的源码（逐文件 SHA-256），先做 CPU 测试，再运行 GPU 子进程。无需额外上传 R1–R59 Notebook。

运行后发送：`SHV_R60_LC_Q31_A0_RESULT_UPLOAD_TO_CHATGPT.zip` 与同名 `.zip.sha256`。
