# Profiling 04 — TinyStories-260K

本实验将一个 batch 从随机权重开始完整训练 4 epoch 记为一个 task。每个 batch 依次完成 30 个 task。

| 物理 GPU | Batch size | Task 数 | 每个 task 的 epoch 数 |
|---|---|---|---|
| 0 | 32 | 30 | 4 |
| 1 | 64 | 30 | 4 |
| 2 | 128 | 30 | 4 |
| 3 | 256 | 30 | 4 |
| 4 | 512 | 30 | 4 |

Task 1–30 的种子为 1337–1366。同编号 task 在五个 batch 下共用相同初始化种子和每轮序列排列，梯度累积为 1。task 之间重新初始化模型、AdamW 状态、学习率进度及随机数状态。每组独立推进。

采用 Karpathy 发布的 stories260K 配置及 llama2.c 源码版本 350e04fe35433e6d2941dce5a1f53308f87058eb。模型为 Llama 2 风格 Transformer，实际参数量 260,032，dim=64、5 层、8 个 query heads、4 个 KV heads、multiple_of=4、dropout=0.05、词表 512、序列长 512。官方模型源码与 MIT 许可证保存在 upstream 目录。[作者发布的配方](https://github.com/karpathy/llama2.c/blob/350e04fe35433e6d2941dce5a1f53308f87058eb/doc/stories260K.md)

完整语料压缩包的 SHA-256 为 26cf7605aca15bc4ea6fa637256400d9d01317b28ed296172b2d1dd160cd7699，发布的 tokenizer 哈希为 dfff07d929db979913f166ec94a6f5ecad4c70cfed8eb5c9cbe7e464455e46f5。按作者的预处理方式，每条故事 strip 后加 BOS、不加 EOS，逐分片按 512 分段，采用 floor(token_count/512)−1 个训练窗口。第 0 分片完整保留作验证，其余 49 分片包含 3,541,812 个窗口、1,813,407,744 个预测 token。100,000 次官方更新、batch 128 对应 3.613969 个 epoch，本实验统一向上取整至 4。

正式运行一次遍历每个训练窗口恰好一次，包含末尾不足一个 batch 的更新。各轮的全量窗口排列由 NumPy SeedSequence([task_seed, one_based_epoch]) 决定，跨 batch 共用，排列哈希写入每个 epoch 行。

AdamW 沿用学习率峰值 0.001、betas=(0.9,0.99)、weight decay=0.01、gradient clip=1.0、bf16 autocast 与 torch.compile。预热按作者 1,000×128 条序列对应的处理量换算，即 128,000 条序列，约 0.03614 epoch。之后以已处理序列数进行 cosine 衰减，到 4 个完整 epoch 时学习率为 0。作者原实现的 min_lr=0，本实验沿用此值。将停止条件、预热与衰减统一到数据进度是本实验的设定。

每个 task 在训练前以及每轮结束时评估完整验证分片，并在固定的 12,800 条训练序列上以 eval 模式计算 train_probe_loss。这些样本由 probe_seed=20261118 选出，在所有 task 和 batch 间共用。train_loss 为训练过程中的整轮平均 CE，train_probe_loss 为固定权重下的训练样本 CE。两种值均单独记录。

正式训练窗口之外完成预编译，并单独记录 compile_warmup 的功率。每 0.5 秒连续采集 GPU 功率、温度、时钟及累计能耗计数器。能耗优先使用 NVML 累计计数器差值，同时记录功率积分以供交叉核查。单独保存训练、验证、训练探针的时长与能耗，epoch_seconds 和 epoch_energy_j 为三部分之和，checkpoint 写盘时间保留在原始功率轨迹中。功率文件中的 block_id 表示 task 编号。

每个 task 保存 initial.json、baseline.json、checkpoint.pt、epochs.csv 和 complete.json。初始权重哈希、每轮数据排列哈希、种子与任务编号使不同 batch 的配对关系可核验。checkpoint 每 300 秒及每轮结束时原子保存，包含训练累积指标、数据位置、权重、优化器与 CPU/CUDA RNG 状态。任务完成后保留最终 checkpoint。

启动脚本仅在映射的物理 GPU 没有计算进程且显存/利用率为空闲状态时发任务。NVML 每次功率采样都会检查是否有其他计算进程，发现共享则记录 overlap 文件并退出，调度器在该卡空闲后恢复最近的 checkpoint。失败后的未提交训练片段保留在原始 telemetry 中，正式累计指标只包含已提交的测量片段。

服务器目录为 /path/to/experiments/tinystories-profile-04。进入 code 目录后可执行

    python -m stories260k.status /path/to/experiments/tinystories-profile-04

配置保存在 config.json，准备数据使用 prepare.py，启动使用 start.sh。全部原始功率位于各 batch 的 power_*.csv.gz，正式指标位于 results/batch_*/task_*/epochs.csv。
