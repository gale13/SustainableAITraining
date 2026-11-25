# OpenCEM PV trajectory database — final v1.0.0

这是供 Profiling 04 / Section 5.2、5.3 后续实验调用的本地数据库与 Python 接口。基准为 **inverter 1**，包含 **242 个有记录日期、1,135,611 个唯一观测时间点**；原始日期及不完整日期全部保留。运行时不访问网络、不占用 GPU。

数据库：`src/opencem_pv/data/opencem_pv.sqlite`，约 88.85 MB。

## 直接使用

在本目录运行，无需安装：

```bash
PYTHONPATH=src python examples/use_daily_pv.py
```

或安装到自己的 Python 环境（需要 NumPy、pandas）：

```bash
python -m pip install -e /path/to/experiment-bundle/opencem-pv-final
```

之后在任意目录调用：

```python
from opencem_pv import PVDatabase

with PVDatabase() as db:
    all_days = db.days()                         # 全部 242 天及质量指标
    experiment_days = db.days(recommended_only=True)  # 142 个推荐实验日

    pv = db.load_day("2025-12-26")
    measured = pv.observations()                 # 实测点，含原始值、缺失标记、来源
    trajectory = pv.sample(step_s=1)             # 1 秒网格，直接调用官方插值函数
    power_w = pv.power_at(10 * 3600 + 0.25)       # 当天 10:00:00.25 的功率
    energy_j = pv.energy(10 * 3600, 185.8)        # 从 10:00 开始，185.8 秒内的 PV 能量

    scaled_pv = db.load_day("2025-12-26", scale=0.1)
    scaled_energy_j = scaled_pv.energy(10 * 3600, 185.8)
```

`PVDatabase("/path/to/opencem_pv.sqlite")` 可指定移动后的数据库。所有日期使用 **Asia/Shanghai（UTC+8）**；`start_s`/`duration_s`/`step_s` 单位为秒，功率为 **W**，能量为 **J**，Wh = J / 3600。`scale` 为无量纲乘数，只作用于查询返回的功率与能量，不改变数据库或 `observations()` 返回的实测原值。

命令行也可调用：

```bash
PYTHONPATH=src python -m opencem_pv days --recommended
PYTHONPATH=src python -m opencem_pv day 2025-12-26 --step 1 --output /tmp/pv_2025-12-26.csv
PYTHONPATH=src python -m opencem_pv energy 2025-12-26 --start 36000 --duration 185.8
```

## 数据清洗规则

固定源仓库版本：[5884d253a5267fb240b7a8df6fa9e4d49a905167](https://github.com/OpenCEM-platform/opencem-dataset/tree/5884d253a5267fb240b7a8df6fa9e4d49a905167/data)。所有输入文件 SHA256、Git blob SHA1、来源 URL 保存在 `src/opencem_pv/data/manifest.json`。

1. 按行实际字段数解析 78 列新导出，排除已确认重叠的 82 列旧导出；解决混合表头导致的 PV 列错位。
2. 只取 inverter 1。原有 1,694,324 行，删除 558,713 行完全重复记录，保留 1,135,611 条。完整行哈希与时间戳共同校验，若同时间戳内容冲突则停止构建，不静默挑选。
3. 其中 **1,134,524 条 PV 功率有效、1,087 条缺失**。保留缺失时间点与原始文本，将其可查询功率标为 NULL。支持标记非有限值/负功率；本次未发现这两类。
4. 保留合法的零功率。未归一化、未平滑、未截断正值、未自动放大、未按昼夜补零；实测有效功率范围为 0–2,984 W。
5. 数据库只保存实测点。插值在调用时生成，不计入实测数据量。原始来源文件编号、行号、行哈希、重复次数均可追溯。

时间范围为北京时间 **2025-07-12 21:31:35—2026-04-12 02:29:20**。242 天是“有记录的日期”，不是 242 个完整的 24 小时；没有观测的日期不能调用为真实轨迹。

## 官方插值与缺口处理

直接调用上游 **`opencem.dataset.interpolate_value()`**。未改写该插值函数，也未用自定义插值器代替它；官方 `dataset.py`、`clock.py`、`interfaces.py` 和空的 `__init__.py` 原样包含在 `src/opencem/`。`upstream_manifest.json` 固定其 Git blob 哈希，并通过测试逐文件验证。实际日轨迹还与官方 `PowerSourceDataset.step()` 逐点对照。

官方方法为相邻有效实测点间的**线性插值**，不是预测缺失天气、生成新光伏样本或恢复高频物理波动。[官方 API 文档](https://tongxin.me/opencem/api.html) / [官方代码](https://github.com/OpenCEM-platform/opencem_simulator/blob/main/src/opencem/dataset.py)。

本接口在调用官方函数前补充边界和缺口检查：

- 默认 `max_gap_s=300`：两侧有效实测点相距超过 300 秒时，不插值跨过该缺口。
- 不对首条之前、末条之后外推；不循环重复一天，不跨日期自动续接。
- `sample()` 返回 `quality=observed/interpolated/long_gap/outside_coverage` 和 `support_interval_s`。缺口保留 NaN，不转成零。
- `power_at()` 默认遇缺口抛出 `DataGapError`；需要检查数组时可设 `missing="nan"`。
- `energy()` 若窗口未完全覆盖则报错；`energy_report()` 单独返回**已覆盖部分**的能量、覆盖秒数和缺失秒数，避免把部分能量误作整轮能量。
- 跨午夜的 epoch 必须由调用方显式拆分到相邻真实日期；本接口拒绝越过当天 86,400 秒。

旧的粗粒度日期仍可看实测点。确需严格复现官方“范围内不限制缺口”的行为时，显式设置：

```python
pv = db.load_day("2025-09-03", max_gap_s=None)
```

此时可以在观测范围内跨越长缺口，但不能把所得曲线当作已恢复的真实变化。正式实验建议使用推荐日期和默认检查。

### 能量积分

`pv.energy(start_s, duration_s)` 计算：

\[
E_{PV}=\int_{start}^{start+duration}P_{PV}(t)\,dt.
\]

它对官方线性曲线逐段精确积分，窗口边界的功率直接调用官方函数。使用原始折点与累计积分，不依赖先生成 1 秒网格；支持小数 epoch 时长。1 秒网格主要用于画图或导出。

## 是否满足 Profiling 04 的时间粒度

根据已有 600 条正式 epoch 观测（每 batch 120 条）：

|Batch size|epoch 时长中位数|13 秒采样下每轮约含多少个间隔|30 task × 4 epoch 的平均总时长|
|---:|---:|---:|---:|
|32|638.4 秒|49.1|21.28 小时|
|64|315.5 秒|24.3|10.52 小时|
|128|185.8 秒|14.3|6.19 小时|
|256|184.3 秒|14.2|6.14 小时|
|512|201.0 秒|15.5|6.70 小时|

因此，**采样约 12–14 秒、无长缺口的日期适合用于现有 Section 5.2/5.3 的 epoch 级能耗比较和 V/权重敏感性实验**。官方插值足以提供任意小数时刻查询与时间窗口积分；它并不能证明未观测的真实秒级波动已经恢复。

早期 111 秒采样下，最快 batch 每 epoch 只有约 1.7 个采样间隔；366 秒采样甚至不足 1 个。这些日期保留用于历史回放或粗粒度敏感性分析，不纳入默认正式实验日期集合。每日粒度以 `day_catalog.csv` 为准；10 月存在频率切换，不能把整个 10 月都当作细粒度。

### 142 个推荐实验日

预先固定、只看数据质量的筛选规则：

- 当天有效测量间隔中位数 ≤15 秒；
- 从午夜到下一午夜，全部时间均有前后实测点支撑，且每个支撑间隔 ≤60 秒（包括跨午夜的相邻实测点）。

符合者共 **142 天**，分布在 2025-10-25—2026-04-10 之间，不表示期间每一天都入选。所有 242 天仍在数据库中。采用默认 300 秒缺口限制时，全天完整覆盖的日期有 195 天，但其中包含粗粒度日期，因此不全部推荐给正式实验。

选择日期与阈值不依赖 LYA 的表现。PV scale 是后续实验参数，数据库不预先按某种工作负载或“70%”进行缩放。

## 接入我们的全量模拟器

```python
# sim 是现有 Profiling 04 全量 Simulator，clock_s 在 task 之间持续累加。
pv = db.load_day("2025-12-26", scale=0.1)
clock_s = 0.0
for task in range(30):
    sim.reset()
    for epoch in range(4):
        result = sim.step(action)
        renewable_j = pv.energy(clock_s, result["T_s"])
        energy_shortfall_j = max(0.0, result["demand_J"] - renewable_j)
        clock_s += result["T_s"]
```

上述短缺量沿用现有实验“每 epoch 总需求减该窗口 PV 总能量再取正”的约定。若以后改为逐秒供需匹配或加入电池，那是另一个建模选择，不应把这两种短缺量混用。

现有 Figure 8/9 的 C++ 加速引擎仍是小时级 PV 输入；本次未静默替换旧实验结果。后续正式运行应使用本接口的能量窗口查询，或显式更新加速引擎为同一分段线性积分规则，不能直接把 1 秒 CSV 塞进原本按小时解释的数组。

## 验证与复现

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/assess_profiling04.py
```

验证包括：全部 242 天可读取、数据库只读、原始值与缺失保留、官方代码哈希、实际轨迹对照官方类、已知解析能量、小数窗口积分、缩放、能量可加性、跨午夜拒绝和长缺口处理。

已有全量模拟器还完成五种 batch、各 30 task 的 **600 epoch 接口冒烟验证**，使用 2025-12-26，所有时间窗口均有完整 PV 支撑。结果见 `examples/profiling04_smoke.csv` 和 `profiling04_assessment.json`。这不是重跑 Figure 8/9，也没有调参。

重建数据库需要已下载的官方原始 CSV 审计目录（本工作区已有）：

```bash
python scripts/build_database.py --audit ../opencem-audit-20261123 --rebuild
```

迁移至另一台机器时，整个本目录即可离线查询；无需携带约 897 MB 的原始 CSV。若需重建，按 manifest 中的来源和 hash 重新获取原始文件，并使用相同审计目录结构。

`pyproject.toml` 提供包安装；`src/opencem_pv/data/manifest.json`、`upstream_manifest.json`、`checksums.json` 记录数据和代码版本；`day_catalog.csv` 提供全部日期的质量指标；`profiling04_resolution.csv` 提供时长对照。上游代码归 OpenCEM 作者所有，详见 `UPSTREAM_NOTICE.md`。
