# 八张 RTX 5090 的持续训练 profiling 实验

学校服务器的普通用户可以直接使用 [Miniconda 安装与 tmux 实验手册](README_NATIVE.md)。

通过 Termius 连接 Ubuntu 服务器，用一个后台 Docker 容器启动八个独立的 PyTorch 进程。每个进程独占一张卡，按顺序读取相同的 WikiText-103 数据块。下一块沿用该卡的模型、Adam 状态、更新步数与随机数状态。下面的命令都在 **Termius 打开的服务器终端** 执行，本机只需用 SFTP 传文件。

## 1. 上传、构建、启动

1. 在 Termius 打开已保存的服务器 Host，再打开该连接的 SFTP。把交付的 `transformer_profiling-src.tar.gz` 和 `transformer_profiling-src.tar.gz.sha256` 上传到服务器的用户主目录。以下示例从空的工作目录开始。
2. 在服务器终端执行：

```bash
cd ~
sha256sum -c transformer_profiling-src.tar.gz.sha256
mkdir -p ~/wt103-lab
tar -xzf transformer_profiling-src.tar.gz -C ~/wt103-lab
cd ~/wt103-lab/transformer_profiling

docker version
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
df -h .
docker build -t sustainable-transformer:cu128-v2 .
bash scripts/container.sh python -m profiling.launch preflight --eight-5090s
```

核对 8 张 5090 都可见且有有效功率读数。镜像固定 Python 3.11、PyTorch 2.7.1 的 CUDA 12.8 构建及项目依赖；主机驱动 580.105.08 提供 GPU 访问，容器通过 NVIDIA Container Toolkit 获得计算及只读监测接口。首次构建需要访问 Debian、PyPI 和 PyTorch 源，首次备数需要访问 Hugging Face；第 5 节给出已有镜像与数据时的离线复用步骤。请为镜像、构建缓存、数据缓存、镜像导出和实验输出预留磁盘空间。若 Docker 报 GPU driver selection 错误，需由服务器管理员检查 Toolkit 的 Docker runtime 配置。[PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/) · [官方 cu128 安装版本](https://pytorch.org/get-started/previous-versions/) · [NVIDIA 容器自检](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)

首次构建完成后启动实验。记住选用的 RUN_ID，后面重连时仍使用这个值。

```bash
RUN_ID=wt103-5090-01
bash scripts/start-server.sh "$RUN_ID"
docker logs --tail 40 -f "wt103-$RUN_ID"
```

看到容器 ID 即表示 Docker 已接受后台任务。按 **Ctrl+C** 退出日志查看，或直接关闭 Termius。若用 `docker exec -it "wt103-$RUN_ID" bash` 进入容器，输入 `exit` 即可退出这个交互 shell。启动脚本使用 `docker run -d --init --restart unless-stopped`，实际训练由容器入口启动。Docker 守护进程重启后会重新启动此前运行的容器。实验完成或发生可诊断错误时，容器保持空闲以便查看结果，GPU 训练进程释放显存。请用 `status.json` 判断实验状态。[Docker 后台运行](https://docs.docker.com/reference/cli/docker/container/run/) · [Docker 重启策略](https://docs.docker.com/engine/containers/start-containers-automatically/)

如需实验结束后自动向 GitHub 提交，现在按第 6 节在服务器宿主机启动等待发布的命令。

## 2. 重连、查看进度与恢复

重新打开同一 Host 的 Termius 终端：

```bash
cd ~/wt103-lab/transformer_profiling
RUN_ID=wt103-5090-01
docker ps -a --filter "name=wt103-$RUN_ID"
cat "results/$RUN_ID/status.json"
tail -n 30 "results/$RUN_ID/coordinator.log"
```

校准完成会生成计划；正式训练开始后可以进一步查看：

```bash
cat "results/$RUN_ID/plan.json"
cat "results/$RUN_ID/training/progress.json"
tail -n 5 "results/$RUN_ID/training/batch_010/worker.log"
```

状态依次为 `preflight`、`preparing_data`、`calibrating`、`training`、`auditing`。正式训练中，`common_completed_blocks` 是八卡均已落盘的块数。主动暂停和继续分别执行下列两条命令：

```bash
docker stop "wt103-$RUN_ID"
# 需要继续时执行
docker start "wt103-$RUN_ID"
```

重启沿用相同的 RUN_ID、镜像、配置快照、数据校验值和 GPU UUID 映射。已完成的共同数据块保持提交状态，训练从最后一个共同块边界恢复。正在训练的块从其第一个 epoch 重算，先前已写入的逐 epoch 记录保存在 `attempts/` 供追溯。校准中断则重新校准，旧尝试保留。24 小时截止时间从本轮容器首次进入实验程序算起，包含备数、校准、停机等待和正式训练；镜像构建与上传在该时钟之外。

`failed` 时先查看上面的协调器日志及对应 batch 的 `worker.log`。修复临时网络、GPU 占用或磁盘问题后，执行 `docker restart "wt103-$RUN_ID"` 重试。八卡应在整场实验中专用。若失败原因为 OOM 或超过显存余量，将 `configs/default.json` 的 `batches` 改成较保守的八档，例如 `[4,6,8,10,12,16,24,32]`，换一个 RUN_ID 用第 1 节启动，重新测量全部八档。已开始的实验使用冻结的配置。若长期停机使时间预算耗尽，可给下一轮使用新的 RUN_ID。若要在服务器验证代码自带的单元测试，可运行 `bash scripts/container.sh python -m unittest discover -s tests`。

## 3. 实验设计与 batch、epoch 的选择

**模型和基准。** 训练任务为逐词的下一个 token 预测。每卡一个 6 层因果 Transformer，宽度 256、FFN 1,024、4 头、序列长 512、dropout 0.3。实现使用正弦位置编码、post-LN、ReLU、输入输出共享词嵌入和完整词表 softmax，固定 32,768 词表时为 13,127,168 个可训练参数。八卡同 seed，并核对初始权重 SHA-256。默认 FP32，禁用 TF32，使用 Adam、梯度范数裁剪 1.0。学习率在每卡最初 100 次真实 optimizer 更新中线性升至 0.0005，之后维持峰值，跨块延续。每个自然 batch 做一次更新，尾批保留。不同 batch 因而会有不同的更新次数、dropout 抽样和优化轨迹。原文的 1.2 小时是单张 GTX 1080 Ti 上的 WT2 Transformer 结果。[论文附录表 5–7](https://aclanthology.org/2021.naacl-main.407.pdf) · [作者实现](https://github.com/SimengSun/revisit-nplm)

**数据和到达顺序。** 准备程序解析并记录 Salesforce/wikitext 的同一提交 SHA，读取 raw 版词串，保留大小写，按空白切词并在每一行末尾加入一个 EOS。采用 raw 版可保留发布者词表原会替换的词，按本实验的首块词表统计 OOV。用同一规则实数 WT2 train 的 token 数，向下对齐至 512 的倍数，得到每块预测目标数 T（约 200 万）。将 WT103 train 切为连续等长块；相邻块复用一个上下文 token，每个预测目标只属于一个块。用首块可见的 T+1 个词建词表，含两个保留项 `<unk>` 和 `<eos>`，词频并列时按词串排序。后续块冻结词表，`unknown_fraction` 为各块预测目标中编码为 `<unk>` 的比例。八卡读取同一份通过哈希校验的数据，块内样本顺序固定。所有卡完成当前块后才同时开放下一块；这是按顺序到达的模拟。余下未启用的完整块和不满一块的尾部均不进入正式训练。验证使用固定的官方 validation split，按完整 512 序列评分全部目标，独立计时计能；此评分口径与原文评估时的 128-token scored target 不同。[数据卡及 raw 版本说明](https://huggingface.co/datasets/Salesforce/wikitext) · [原文评估设置](https://aclanthology.org/2021.naacl-main.407.pdf)

**候选显存范围。** 32,607 MiB/卡约为 31.84 GiB。参数、梯度和 Adam 两份 FP32 状态约 200 MiB。一个词表 logits 张量需 B × 512 × 32,768 × 4 字节，即每增加一条序列多 64 MiB。下表只估算单份 logits；完整峰值还包含交叉熵中间张量、反向梯度、六层激活、注意力及 CUDA 工作区。

| GPU 索引 | batch（序列/更新） | 满批 tokens/更新 | 单份 FP32 logits |
|---:|---:|---:|---:|
| 0 | 4 | 2,048 | 0.25 GiB |
| 1 | 8 | 4,096 | 0.50 GiB |
| 2 | 10 | 5,120 | 0.625 GiB |
| 3 | 16 | 8,192 | 1.00 GiB |
| 4 | 24 | 12,288 | 1.50 GiB |
| 5 | 32 | 16,384 | 2.00 GiB |
| 6 | 48 | 24,576 | 3.00 GiB |
| 7 | 64 | 32,768 | 4.00 GiB |

B=10 对齐论文基准。八卡先同时做真实前向、反向、Adam 更新以及验证，每卡完成至少三个完整 pilot epoch、至少 180 秒校准；此前还有约 30 秒内核预热。程序检查 PyTorch 分配器的峰值 reserved 显存占设备总显存至多 85%（该硬件约 27.07 GiB）。每卡的 `pilot.json` 记录实际 allocated/reserved 峰值、设备总显存及逐 epoch 用时。正式实验重新初始化权重和 Adam，校准训练不计入正式曲线。

**所有 batch 共用一个 E 和一个块数。** 论文基准上限相当于 10,000 × 5,120 = 51,200,000 个训练目标；对约 200 万目标的块，起点约 25 个 epoch。设 t_fast 是八卡 pilot 中最短的单个训练 epoch 秒数，t_slow 是最长的单个“训练＋验证”epoch 秒数。机器按下式定稿并写入 `plan.json`：

```text
E_paper = ceil(51,200,000 / T)
E       = max(E_paper, ceil(1,200 × 1.20 / t_fast))
块预算   = E × t_slow × 1.15
K       = min(可用完整块数, floor((24 小时 - 已耗时 - 30 分钟预留) / 块预算))
```

固定 E 后，八卡每个正式块都训练 E 遍，每卡每个 epoch 都预测 T 个目标。最快卡按 24 分钟纯训练量规划，给 20 分钟目标留有余量。最慢卡决定能共同完成的块数；每次开启新块前还会参考上一块的实际墙钟耗时，剩余预算不足便共同结束。最后一块允许完整跑完，实际总时长受散热、时钟和 I/O 影响。每个已完成块都用实测数据检查 1,200 秒下限，实际最短值见 `audit.json`。若状态为 `complete_with_duration_shortfall`，结果完整可导出，同时 `block_summary.csv` 标明低于目标的卡和块。需要重新满足下限时，可据观察到的最快 epoch 增大 `minimum_duration_margin`，换 RUN_ID 重跑全部八卡。

## 4. 读结果、能耗与编号

所有输出在服务器宿主机 `results/<RUN_ID>/`，退出、停止或移除容器后仍在该目录。正式结果位于 `training/`。例如 batch=10、首块：

```text
results/wt103-5090-01/
  status.json                 实验状态，结束时含实际总时长
  job.json, config.json       镜像 ID、启动时间和配置快照
  coordinator.log, plan.json  阶段日志和实测校准计划
  pilot/<attempt>/            校准原始测量、各 batch 的 pilot.json 和设备环境
  training/
    data_manifest.json        数据版本、切分、哈希及各块未知词比例
    run.json, progress.json   设备映射、共同提交块数、提前结束原因
    batch_010/
      block_0000/             metrics.csv、loss.csv、time.csv、energy.csv
      telemetry/              GPU 原始功率、时间戳、UUID、阶段、块和 epoch
      attempts/               每个 epoch 原子更新一次的逐次尝试记录
      worker.log              逐 epoch 日志
    all_epochs.csv            全部共同完成块的逐 epoch 总表
    block_summary.csv         每卡每块的实测时间、能耗、20 分钟检查
    audit.json                核对编号、token 暴露量、步数连续性、时长和能耗
```

三张分表首列均为 `epoch_id`。例如 `b0000_e00001` 表示块 0 的第 1 个 epoch；在同一 batch 的三张表按它关联。在跨卡、跨实验的总表中，使用 `(run_id, batch_sequences, epoch_id)` 作为联合键。训练 loss 是当次全部目标的 token 加权平均交叉熵；验证 loss 在该 epoch 更新完成后评估。用同一条命令即可在服务器读取三个指标：

```bash
python3 - "$RUN_ID" <<'PY'
import csv, pathlib, sys
p = pathlib.Path("results") / sys.argv[1] / "training/batch_010/block_0000/metrics.csv"
with p.open() as stream:
    row = next(r for r in csv.DictReader(stream) if r["epoch_id"] == "b0000_e00001")
print({k: row[k] for k in ("epoch_id", "train_loss", "train_seconds", "train_energy_j", "unknown_fraction")})
PY
```

`train_seconds` 与 `train_energy_j` 对应同一个经过 CUDA 同步的训练窗口；验证各有独立时长与能耗。每 epoch 的 `epoch_wall_seconds` 含两个阶段和边界开销；`epoch_energy_j`、`epoch_energy_wh` 是两个测量窗口的和。能耗单位为焦耳与瓦时，1 Wh = 3,600 J。优先使用按 GPU UUID 读取的 NVML 累计毫焦计数器差；不可用或停滞时，使用约 0.5 秒一次、按真实时间戳采样的板卡功率做梯形积分，同时保存边界采样。NVML 功率读取初始化失败则尝试只读的 nvidia-smi。两种接口均不可读或积分有严重缺样时，程序报告失败。功率读数在部分 NVIDIA 卡上经过约一秒平滑，短验证窗口应结合 `sample_count`、`max_sample_gap_s` 与 `sensor_note` 解读。报告覆盖所选 GPU 板卡，CPU、主机风扇、电源损耗和块间等待均在测量口径之外。[NVIDIA NVML 设备计量接口](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)

断点在 `batch_*/_state/` 临时滚动保存模型、Adam 和 RNG，以便真正续训。结束并通过数据完整性核对后程序清理这些断点，保留全部采集输出和原始遥测。可用 Termius SFTP 下载整个 `results/<RUN_ID>/` 或第 6 节归档。

## 5. 保存 Docker 镜像、重新上传与离线复用

首次构建后在服务器执行一次，镜像归档可从 Termius SFTP 下载：

```bash
cd ~/wt103-lab/transformer_profiling
mkdir -p images
docker image save -o images/sustainable-transformer-cu128-v2.tar sustainable-transformer:cu128-v2
(cd images && sha256sum sustainable-transformer-cu128-v2.tar > sustainable-transformer-cu128-v2.tar.sha256)
```

镜像包含代码和 Python/CUDA 依赖。为了下次使用同一份数据，等程序进入校准或训练后，可以单独备份已经完整发布的数据目录：

```bash
tar -czf images/wt103-prepared.tar.gz -C data wt103-raw-first32k-s512-v1
(cd images && sha256sum wt103-prepared.tar.gz > wt103-prepared.tar.gz.sha256)
```

在另一台已配置 NVIDIA Container Toolkit 的兼容 Linux x86-64 GPU 服务器上，通过 Termius SFTP 上传源码包、镜像 tar、数据包及其校验文件。按第 1 节解开源码包，将后四个镜像/数据相关文件放到项目的 `images/` 中，然后运行：

```bash
cd ~/wt103-lab/transformer_profiling
(cd images && sha256sum -c sustainable-transformer-cu128-v2.tar.sha256 && sha256sum -c wt103-prepared.tar.gz.sha256)
docker image load -i images/sustainable-transformer-cu128-v2.tar
mkdir -p data
tar -xzf images/wt103-prepared.tar.gz -C data
RUN_ID=wt103-5090-02
bash scripts/start-server.sh "$RUN_ID"
```

有网络时也可只上传镜像与源码包、跳过数据校验和解包，程序将自动备数。新 RUN_ID 使用新模型重新开始；要续接同一轮中断实验，还需该轮宿主机的完整输出及断点，并满足原 GPU UUID 映射与剩余时间预算。若只要重新运行镜像内置配置，下面是省去源码包的等效启动命令（GitHub 发布工具仍在源码包中）：

```bash
# 在希望保存实验数据和结果的服务器工作目录中执行，镜像需已 load
RUN_ID=wt103-5090-03
IMAGE_ID=$(docker image inspect --format '{{.Id}}' sustainable-transformer:cu128-v2)
mkdir -p data cache results
docker run -d --init --restart unless-stopped --name "wt103-$RUN_ID" \
  --gpus all --shm-size 2g --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  -e HOME=/tmp -e "PROFILE_RUN_ID=$RUN_ID" -e "PROFILE_IMAGE_ID=$IMAGE_ID" \
  -v "$PWD/data:/data" -v "$PWD/cache:/cache" -v "$PWD/results:/results" \
  "$IMAGE_ID" python -m profiling.pipeline
```

镜像、准备好的数据与实验结果各自独立保存；用 `docker image save/load` 复用带标签的镜像，用 SFTP 或 GitHub 备份挂载目录中的结果。[Docker image save](https://docs.docker.com/reference/cli/docker/image/save/) · [Docker image load](https://docs.docker.com/reference/cli/docker/image/load/)

## 6. 在宿主机同步结果至 GitHub

目标仓库是 [gale13/SustainableAITraining](https://github.com/gale13/SustainableAITraining)。宿主机需要 Python 3.9+、git、该仓库的写权限和可用于非交互推送的 HTTPS 凭据。若已安装 GitHub CLI，可以在 Termius 中执行 `gh auth login --hostname github.com --git-protocol https --web` 完成设备授权，再运行 `gh auth setup-git`。认证发生在宿主机，训练容器无需 GitHub 凭据。

以下命令在服务器项目目录执行。首次使用时 clone 一份专门的结果仓库；若已 clone，直接使用原路径。请确认 git 已配置你的提交姓名和邮箱。

```bash
cd ~/wt103-lab/transformer_profiling
RUN_ID=wt103-5090-01
RESULT_REPO="$HOME/wt103-lab/SustainableAITraining"
git clone https://github.com/gale13/SustainableAITraining.git "$RESULT_REPO"
git -C "$RESULT_REPO" switch -c "codex/profiling-$RUN_ID"
git -C "$RESULT_REPO" var GIT_AUTHOR_IDENT
nohup env GIT_TERMINAL_PROMPT=0 python3 -u scripts/sync_results.py \
  --run "$PWD/results/$RUN_ID" --repo "$RESULT_REPO" --wait --push \
  > "$PWD/results/publish-$RUN_ID.log" 2>&1 < /dev/null &
```

发布程序等待采集结束，向该分支提交 `profiling-results/<RUN_ID>/`。GitHub 中的 `tables/` 直接展示分块 CSV 和总表。完整归档还含原始遥测、尝试记录、校准、配置、数据清单和日志，按 40 MiB 分卷并提供 SHA-256；模型权重、语料和 Docker 镜像均从归档中排除。若 20 分钟目标有偏差，审核标记与完整结果一起发布。查看同步状态：

```bash
cat "results/publish-$RUN_ID.log"
git -C "$RESULT_REPO" ls-remote --heads origin "codex/profiling-$RUN_ID"
```

若服务器清理登出用户的后台 shell 进程，或认证中断，待完成后在重连的终端重新运行下式；它会复用同一目的目录与已有提交，重试推送：

```bash
python3 scripts/sync_results.py --run "results/$RUN_ID" --repo "$RESULT_REPO" --push
```

若服务器无法访问 GitHub，可在服务器完成导出，再用 Termius SFTP 下载导出目录；在有仓库访问权限的机器上，将该目录放入目标仓库的 `profiling-results/<RUN_ID>/`，按一般 git 流程提交、推送即可。导出命令：

```bash
python3 scripts/sync_results.py --run "results/$RUN_ID" --export-dir "$HOME/export-$RUN_ID"
```

在下载的导出目录或仓库分支的 `profiling-results/<RUN_ID>/` 中恢复完整文件：

```bash
sha256sum -c SHA256SUMS
cat results.tar.gz.part* > results.tar.gz
sha256sum -c archive.sha256
tar -xzf results.tar.gz
```
