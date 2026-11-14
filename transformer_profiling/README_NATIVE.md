# 学校服务器部署：Miniconda 环境与 tmux 会话

通过 Termius 连接学校服务器。以下安装命令在服务器执行；创建训练会话时，请位于**学校分配给你的同一台八卡 RTX 5090 计算节点**。会话与服务器进程相连，普通 Termius 断线或主动 detach 后可以继续。学校若在登出时清理用户进程，或交互式 GPU 配额到期，任务仍会受该策略影响。[tmux 官方手册](https://man.openbsd.org/tmux)

## 1. 部署 Miniconda

用 Termius SFTP 把 `transformer_profiling-conda-tmux.tar.gz` 及其 `.sha256` 文件上传到服务器主目录，再在 Termius 终端执行：

```bash
cd ~
sha256sum -c transformer_profiling-conda-tmux.tar.gz.sha256
mkdir -p ~/wt103-conda
tar -xzf transformer_profiling-conda-tmux.tar.gz -C ~/wt103-conda
cd ~/wt103-conda/transformer_profiling
bash scripts/install-miniconda.sh
"$HOME/miniconda3/bin/conda" --version
```

脚本从官方仓库下载固定的 Linux x86-64 Miniconda 安装包，校验 SHA-256 后，通过安装器的 `-b -p "$HOME/miniconda3"` 选项安装到个人目录。已有可用的 `~/miniconda3` 时直接复用。整个流程无需 sudo，也无需修改 shell 启动文件。若使用项目盘，可把项目放到计算节点能访问的路径，并在每次操作前设置 `export PROFILE_CONDA_ROOT=/你的绝对路径/miniconda3`。手册后面的示例使用默认位置。[Miniconda 官方安装包与校验值](https://repo.anaconda.com/miniconda/)

如果只有浏览器能下载，可以从官方索引下载 **Miniconda3-py311_25.9.1-3-Linux-x86_64.sh**，再经 Termius SFTP 上传至服务器主目录。在项目目录用下面的命令代替上面的安装脚本调用；仍会核对同一个 SHA-256。

```bash
PROFILE_MINICONDA_INSTALLER="$HOME/Miniconda3-py311_25.9.1-3-Linux-x86_64.sh" bash scripts/install-miniconda.sh
```

## 2. 创建并验证独立 Conda 环境

```bash
cd ~/wt103-conda/transformer_profiling
bash scripts/setup-native.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/wt103-profile"
python --version
which python
tmux -V
```

安装脚本通过 Conda 创建名为 `wt103-profile` 的环境，使用 conda-forge 安装 Python 3.11、pip 和 tmux；再用**该环境中的 Python**安装 PyTorch 2.7.1+cu128 及采集依赖。输出的 Python 路径应结束于 `miniconda3/envs/wt103-profile/bin/python`。自检会打印 `CUDA build 12.8`。创建环境可在能联网的服务器节点完成，计算节点需能访问同一安装目录。学校有指定 Conda 镜像时，可在运行脚本前设置 `PROFILE_CONDA_CHANNEL` 为学校给出的频道 URL。[PyTorch 官方 cu128 发行版本](https://pytorch.org/get-started/previous-versions/)

## 3. 检查 GPU，创建 tmux 会话

在已分配的计算节点上，进入上述项目目录并激活环境。记录节点名，重连时用得到。

```bash
hostname
python -m profiling.launch preflight --eight-5090s
```

应看到 **8 条 5090 设备与功率记录**，末尾为 `Preflight telemetry OK`。手动自检可在正式占用 GPU 前先发现设备数量或功率读取权限的问题。然后创建后台会话：

```bash
bash scripts/start-tmux.sh wt103-01
```

`wt103-01` 是本轮 RUN_ID，也是专用 tmux socket 的名称；socket 内的会话名是 `train`。脚本调用 `tmux -L wt103-01 new-session -d -s train`，把工作目录设为项目目录，并在会话中执行 `bash scripts/start-native.sh wt103-01`。训练入口会自行激活 Conda 环境，启动信息和错误会写入 `results/wt103-01.tmux.log`。对已存在的同名活动会话重复执行启动命令，只会打印连接方式。

**启动脚本返回终端提示符就可以关闭 Termius。** 如果希望先看一眼八卡试跑，可以连接会话：

```bash
tmux -L wt103-01 attach -t train
```

查看后，按 **Ctrl+B**，松开两个键，再按 **D**，即可分离会话。训练会先自动试跑，根据实测速度和显存生成计划，再重新初始化八套模型和优化器并自动正式训练。正式训练跨数据块延续每张卡自己的状态。

## 4. 重连、查看结果与恢复

用 Termius 重连到**原计算节点**。新终端可以直接用环境内 tmux 的绝对路径：

```bash
cd ~/wt103-conda/transformer_profiling
"$HOME/miniconda3/envs/wt103-profile/bin/tmux" -L wt103-01 ls
"$HOME/miniconda3/envs/wt103-profile/bin/tmux" -L wt103-01 attach -t train
```

如果只想查看进度，可直接读文件：

```bash
cd ~/wt103-conda/transformer_profiling
tail -n 30 results/wt103-01.tmux.log
cat results/wt103-01/status.json
```

`calibrating` 表示试跑，`training` 表示正式训练。进入训练后可以读取 `results/wt103-01/plan.json`；每块的 loss、时间与能耗 CSV 位于 `training/batch_*/block_*/`。作业结束后 tmux 会话会退出，用 `status.json` 和 `training/audit.json` 判断完成情况。若状态为 `failed`，或启动时会话很快消失，查看上面的 tmux 日志及 `results/wt103-01/coordinator.log`。

进程中断后，要从共同块断点恢复，在同一项目、同一 Conda 环境和原八卡 UUID 映射下，再执行 `bash scripts/start-tmux.sh wt103-01`；当前未完成的数据块会重新计算。需要发起全新实验时换 RUN_ID。单轮 24 小时预算从实验程序首次开始算起，包含中断期间。如果学校要求 Slurm 托管，可以在 tmux 中提交随附的 `sbatch scripts/profile.sbatch wt103-01`，按学校要求补充分区或账户参数；该批作业使用同一 Conda 训练入口。[Slurm 批作业](https://slurm.schedmd.com/sbatch.html)

## 5. 保存环境与实验结果

在项目目录中激活该环境后可以保存包版本：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/wt103-profile"
conda list --explicit > conda-linux-64.txt
python -m pip freeze > python-packages.txt
```

完整实验输出已保存在 `results/wt103-01/`，可以直接通过 SFTP 下载。完成后也可打包成带校验值的归档：

```bash
python scripts/sync_results.py --run results/wt103-01 --export-dir "$HOME/export-wt103-01"
```

如已有目标 GitHub 仓库的可写克隆，在下面的命令中填入克隆位置即可发布。实验数据、原始遥测和日志会归档，正式训练的临时模型断点会在审核完成后清理。[GitHub 同步和恢复说明](README.md#6-在宿主机同步结果至-github)

```bash
python scripts/sync_results.py --run results/wt103-01 --repo "$HOME/wt103-lab/SustainableAITraining" --push
```

## 6. 计算节点不能联网时

在能联网、且与计算节点共享项目目录和 Miniconda 安装的校园节点上完成前两节，并提前准备数据。若已有完整的同名数据目录，直接复用即可。首次备数时，在项目目录激活 `wt103-profile` 后运行：

```bash
mkdir -p data cache/huggingface
staging=$(mktemp -d "$PWD/data/.prestage-XXXXXXXX")
HF_HOME="$PWD/cache/huggingface" python -m profiling.prepare --out "$staging/prepared" &&
  mv -T "$staging/prepared" data/wt103-raw-first32k-s512-v1
```

环境和这份数据均已就绪后，训练可以直接在计算节点读取；八卡自检和试跑在实际训练节点上执行。
