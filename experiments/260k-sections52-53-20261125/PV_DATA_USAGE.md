# 项目 PV 数据调用规范

生效日期：2026-11-23。当前固定版本：`opencem-pv-trajectories 1.0.0`。

**本项目后续所有涉及 PV 的新实验、模拟器接入、策略比较和绘图，统一通过 `opencem_pv.PVDatabase` 调用 `opencem-pv-final` 中的数据。** 本文是项目级使用规范；完整 API、清洗细节与验证记录见[接口包说明](opencem-pv-final/README.md)。

历史实验保留原始结果及来源说明。重现历史结果时可以保留当时的数据处理方法，但必须标明“历史复现”；不能把旧数据路径、小时均值或循环轨迹作为新实验的默认入口。

## 1. 唯一数据入口与版本

| 项目 | 固定内容 |
| --- | --- |
| Python 入口 | `from opencem_pv import PVDatabase` |
| 接口目录 | `opencem-pv-final/` |
| 默认只读数据库 | `opencem-pv-final/src/opencem_pv/data/opencem_pv.sqlite` |
| 基准设备 | inverter 1，PV 功率 `pv1power` |
| 日期范围 | 北京时间 2025-07-12 21:31:35 至 2026-04-12 02:29:20 |
| 观测规模 | 242 个有记录日期；1,135,611 个唯一观测时间点，其中 1,134,524 个功率值有效 |
| 正式实验默认日期池 | `db.days(recommended_only=True)` 返回的 142 天 |
| 时区和单位 | Asia/Shanghai；时间 s、功率 W、能量 J；Wh = J / 3600 |
| 上游数据版本 | `5884d253a5267fb240b7a8df6fa9e4d49a905167` |

242 天不等于 242 个完整日。推荐日期位于 2025-10-25 至 2026-04-10 之间，以接口返回的实际名单为准，不自行生成连续日期范围。其筛选仅依据质量：有效采样间隔中位数 ≤15 秒，全天均有间隔 ≤60 秒的前后实测点支撑，包括午夜边界。

原始重复已清理，缺失、合法零功率和来源记录均保留。日常实验不再次解析上游混合 CSV、不另建清洗副本、不直接查询 SQLite 来绕开接口。版本依据保存在[数据 manifest](opencem-pv-final/src/opencem_pv/data/manifest.json)、[官方代码 manifest](opencem-pv-final/upstream_manifest.json)和[文件校验表](opencem-pv-final/checksums.json)。

## 2. 安装与最小调用

在项目根目录执行一次，安装到运行实验所用的 Python 环境：

```bash
python -m pip install -e ./opencem-pv-final
```

安装后可在任意目录导入。运行时使用本地数据，无需联网或 GPU。

```python
from opencem_pv import PVDatabase

with PVDatabase() as db:
    dates = db.days(recommended_only=True)
    date = "2025-12-26"  # 调用示例，不是算法表现最优日的声明
    if date not in set(dates["date"]):
        raise ValueError("正式实验应从推荐日期池选取")

    pv = db.load_day(date, scale=1.0, max_gap_s=300)
    measured = pv.observations()       # 原始实测点，功率不受 scale 影响
    curve = pv.sample(step_s=1)        # 绘图用 1 秒网格；包含质量标记
    power_w = pv.power_at(36000.25)    # 当天 10:00:00.25，单位 W
    energy_j = pv.energy(36000, 185.8) # 10:00 起 185.8 秒内，单位 J
    print(len(dates), power_w, energy_j)
```

`PVDatabase()` 默认定位包内数据库，与当前工作目录无关。移动数据库后可显式传入 `PVDatabase("/path/to/opencem_pv.sqlite")`，并记录该文件的版本和校验值。

命令行示例（安装后）：

```bash
python -m opencem_pv days --recommended
python -m opencem_pv day 2025-12-26 --step 1 --output /tmp/pv_2025-12-26.csv
python -m opencem_pv energy 2025-12-26 --start 36000 --duration 185.8
```

不安装时，从项目根目录将命令写成 `PYTHONPATH=opencem-pv-final/src python ...` 即可。

## 3. 插值、能量、缺口和跨天规则

| 使用目的 | 统一方法 | 含义 |
| --- | --- | --- |
| 查看实测记录 | `pv.observations()` | 不增加插值点，保留原值、缺失及来源 |
| LYA 的当前可观测功率 | 从公共 `observations()` 取时间戳 ≤ 决策时刻的最新有效功率 | 2026-11-25 用户指定；午夜需接入之前的真实观测，不能取插值的午夜功率 |
| 绘制轨迹或导出网格 | `pv.sample(step_s=1)` | 当天 `[0, 86400)` 的插值网格，含 `quality` 和 `support_interval_s` |
| 查询某个时刻功率 | `pv.power_at(seconds)` | 自北京时间当天午夜起的秒数，支持小数和数组 |
| 计算一个 epoch 的 PV 能量 | `pv.energy(start_s, duration_s)` | 对官方分段线性曲线精确积分，支持小数边界 |
| 批量查询能量窗口 | `pv.energy_many(starts_s, durations_s)` | 使用同样的积分和完整覆盖检查 |
| 检查不完整窗口 | `pv.energy_report(start_s, duration_s)` | 报告已知能量、覆盖秒数、缺失秒数及是否完整 |

功率插值直接调用固定版本的官方 `opencem.dataset.interpolate_value()`。**1 秒输出是插值表示，不是 1 Hz 实测，也不代表恢复了未观测的高频波动。** 计算 epoch 能量使用 `energy()`，不先把轨迹变成小时均值，也不以 1 秒网格求和替代已有精确积分。

默认 `max_gap_s=300`。超过 300 秒的支撑间隔不跨越插值；观测范围之外不外推。`sample()` 将缺口保留为 NaN 并标记 `long_gap` 或 `outside_coverage`；`power_at()` 和 `energy()` 默认在缺失窗口报错。不能将缺口补零、无限前向填充，或把 `energy_report().known_energy_j` 当作完整窗口能量。

推荐日期的 60 秒筛选门槛与接口的 300 秒缺口上限是两个不同概念：前者决定默认实验日期池，后者是每次查询的保护规则。历史复现或专门的缺口敏感性分析可以显式设置 `max_gap_s=None`，但必须在实验配置及报告中记录，不能作为默认行为。

`load_day()` 不会自动循环某一天，也不会自动跨天。若一个 epoch 跨午夜，调用方应按**相邻真实日历日期**拆分窗口，分别调用两天的 `energy()` 后相加；相邻日不存在或有缺口时应报告该场景不可用，不能跳到下一条可用日期冒充连续天气。单日实验必须检查每个窗口不超过 86,400 秒。

## 4. 接入 Profiling 04 全量模拟器

正式使用的训练环境为[全量校准模拟器](tinystories-04-simulator-full-data/README.md)。action 为 32、64、128、256、512，每 task 4 个 epoch。PV 时钟按每轮返回的真实模拟时长推进，**在 task 之间持续累加**；`sim.reset()` 只重置训练任务状态。

以下代码在项目根目录运行，演示从北京时间 08:00 开始、固定 batch 128、30 个 task 的接口接法；`scale=0.1` 仅是显式缩放示例，不代表 10% 或 70% 渗透率，也不是建议的实验最优参数。

```python
from pathlib import Path
import sys
from opencem_pv import PVDatabase

sys.path.insert(0, str(Path("tinystories-04-simulator-full-data").resolve()))
from simulator import Simulator

sim = Simulator(seed=42)
clock_s = 8 * 3600.0
rows = []
with PVDatabase() as db:
    pv = db.load_day("2025-12-26", scale=0.1, max_gap_s=300)
    for task in range(30):
        sim.reset()
        for epoch in range(4):
            outcome = sim.step(128)
            duration_s = float(outcome["T_s"])
            if clock_s + duration_s > 86400:
                raise ValueError("超出单日；需显式接入下一真实日期")
            pv_j = pv.energy(clock_s, duration_s)
            shortfall_j = max(0.0, float(outcome["demand_J"]) - pv_j)
            rows.append(dict(task=task, start_s=clock_s, pv_J=pv_j,
                             shortfall_J=shortfall_j, **outcome))
            clock_s += duration_s
print(len(rows), clock_s, sum(row["pv_J"] for row in rows))
```

这个例子沿用现有 Profiling 04 实验的 **epoch 净能量短缺**：`max(demand_J − epoch_PV_J, 0)`。它不同于逐时刻供需差取正后再积分；更换数据入口不会自动改变短缺模型。后续若研究瞬时匹配或储能，应单独记录模型变化。

**2026-11-25 用户更新了 LYA 的信息设置：决策端只用三个 MLP 的 mean，完全不访问噪声库。** 对所有候选 action，先取最近有效实测功率 P_last（按同一 α 缩放），然后计算 `estimated_PV = P_last * m_T`、`estimated_shortfall = max(m_dem - estimated_PV, 0)`。不查询未来时间窗口，不进行决策 PV 积分，也不用可能引用下一条未来观测的 `power_at(t)` 插值值。这个代入值是短缺预测，不是实际短缺的精确条件期望。

实际环境响应仍采用 mean＋经验联合噪声，然后通过 `pv.energy(start, actual_duration)` 计算实际 PV 能量。历史版本曾对所有噪声样本的未来窗口积分后取均值，仅作为旧信息模式保留，不继续作为当前 LYA 的要求。

最近历史样本统一从公共 API 的 `observations()` 获取。午夜之前的样本需要从之前有记录日期的公共观测表取得，不能拿环境插值得到的午夜值代替。当前实现见 [pv_observations.py](260k-section52-mean-observed-pv/pv_observations.py)。没有新样本时持有旧值，并记录 `observation_age_s`；这只是控制器的功率预测方式，不会将环境的缺口填成常数或新增实测数据。

用户进一步将 **MLP mean-only 决策、最近实测功率 × mT、阶段权重 (100,10,1)** 固定为后续默认。Section 5.3 的当前 V 实验使用相同信息规则，唯一 V 图为 [figure9_V_ab](260k-section53-v-mean-observed-pv/figures/figure9_V_ab.pdf)：V–平均 TARS 和 V–平均虚拟队列。旧版本的其他 V 图退出当前图集，保留在历史实验中用于审计；本次不更新权重对比实验。

随后完成的 [固定 V=0.1 权重比较](260k-section53-weight-fixedV/README.md) 也使用完全相同的 PV、控制器信息和环境规则，仅把阶段权重作为对照变量；没有重新选择 PV 倍率、日期或 V。该实验不支持递减权重在此设置下具有总体 TARS 优势，详见完整配对结果。

此前权重图进一步扩展到 [四种 decay 函数与 Fixed/Increasing 的比较](260k-section53-weight-families-fixedV/README.md)。使用相同 240 日期、PV70、真实历史观测预测、环境积分和固定 V=.1，仅新增预先声明的阶段权重函数；PV 规则没有变化。

用户随后放弃固定 V 展示，当前权重图改为 [同 V 下四种 decay 相对 Fixed 的 TARS 降幅](260k-section53-weight-sameV/README.md)。按参考图固定 σe=.235，完整评估 17 个 V（10⁻⁶ 至 10²），其余阶段权重定义、240 日期、午夜起点、PV70 倍率 α=4.1928776597152995、配对种子和上述决策/环境信息规则不变。没有再次调整 PV scale 或选择日期；无限跨缺口官方插值仅延续用户对这一组 240 天实验的明确授权。当前 V 的 a,b 图保持不变。

另按用户要求新增 [同 V 下相对 Increasing 的配套图](260k-section53-weight-sameV-increasing/README.md)，使用 Increasing=(100,1000,10000)，PV 设置与上图完全相同。直接复用四种 decay 的原始运行，仅补算缺失的 Increasing 对照；两个版本并列保留。

**旧加速引擎尚未迁移。** `profiling04-policy-experiments/engine.cpp` 使用小时数组及循环时间轴，不能直接接收 1 秒 CSV。今后启动新 Figure 8/9 实验时，应直接接入本 API；若保留 C++ 加速，则需显式实现与本 API 等价的分段线性积分、缺口及边界行为，并对实际 epoch 窗口做一致性核验后再使用。不得静默沿用旧小时数据或覆盖旧结果的来源说明。

## 5. PV scale 与实验留档

`scale` 是乘在原始功率上的无量纲倍数，同样线性作用于能量；数据库和 `observations()` 原值不变。默认 `scale=1.0`。不能直接继承旧假设负载下的“70% PV”。

每组实验必须明确渗透率的定义、参考负载及时间窗口。例如，若定义为“固定窗口平均 PV 功率 / 固定参考 GPU 平均功率”，则 `scale = 目标渗透率 × 参考平均功率 / 原始窗口平均PV功率`，其中原始窗口平均 PV 功率由 `energy()/窗口秒数` 求得，且必须大于零。这只是定义示例；不要混用平均功率比、总能量比和实际自给率。

同一比较场景内，所有算法使用相同日期、起始时刻及 scale，不按每个算法自身耗能重新缩放。日期和 scale 的搜索应记录候选范围、选择规则与调参/评估用途；按算法表现选择的日期要如实标为调参或探索结果。

每次新实验的配置文件至少保存：

- 数据包版本、上游 commit、实际数据库 SHA256、官方代码 manifest；
- inverter、时区、实际日期列表、起始秒数和日期筛选规则；
- `scale`、渗透率定义与参考负载、`max_gap_s`；
- 插值函数 `opencem.dataset.interpolate_value`、能量方法 `PVDay.energy`；
- 单日或跨天方式、缺口处理结果、短缺模型；
- task 数、每 task epoch 数、随机种子及算法参数。

不要只写“使用 OpenCEM 数据”或“PV=70%”而省略上述定义。更新数据版本时保留旧实验所用的版本记录，并更新本文的固定版本信息。

## 6. 相关文档与已有验证

- [接口包完整说明](opencem-pv-final/README.md)：数据清洗、API、缺口规则及重建方法。
- [全部日期质量目录](opencem-pv-final/day_catalog.csv)：242 天逐日质量记录。
- [Profiling 04 粒度评估](opencem-pv-final/profiling04_assessment.json)：现有 epoch 时长与 PV 粒度的适配情况。
- [已完成验证记录](opencem-pv-final/verification.json)：接口包测试与安装检查。
- [全量模拟器接入冒烟结果](opencem-pv-final/examples/profiling04_smoke.csv)：五种 batch、各 30 task、共 600 个 epoch 的查询验证；不是重新运行 Figure 8/9。

本规范采用已发布的 `v1.0.0` 数据与接口，不改写冻结包、已有模拟器权重或历史实验结果。

## 7. 已完成的新接口策略实验

2026-11-23 的 [260K simulator / Section 5.2 全日期实验](260k-section52-all-days/README.md) 已完成新接口接入。其独立 C++ 加速器对公共 API 提供的原生实测节点和午夜边界值作相同分段线性积分，在全部 240 日的 7,920 个窗口上与 `PVDay.energy()` 核验，最大差约 0.00034 J；模型、噪声与策略轨迹也经原 Python 实现复核。它不是旧 `profiling04-policy-experiments/engine.cpp` 的小时循环实现。

该次实验中，用户明确指定 `max_gap_s=None`，使用 240 个有前后支撑日期的官方无限跨缺口插值，并保留 195/213/142 天质量子集对照。这是已记录的场景特例，不改变本规范为其他新实验规定的默认 `max_gap_s=300`。全部倍率对所有日期统一使用；每次从午夜开始，30 个 task，task 间 PV 时钟持续推进。

Section 5.2 曾先改为[递减权重版本](260k-section52-decreasing-weight/README.md)。**当前版本**进一步按用户 2026-11-25 要求改为 [MLP mean＋最近实测 PV 决策](260k-section52-mean-observed-pv/README.md)。权重固定为 `(100,10,1)`，在原调参日期重选 V，PV70 下仍为 V=0.1。环境 PV、倍率、缺口规则和六个 baseline 保持原设置；所有新 LYA 调参和评估运行均重新计算。旧固定权重及未来积分信息版本保留作历史比较。
