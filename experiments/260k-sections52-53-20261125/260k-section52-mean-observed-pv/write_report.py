"""Report the revised information assumptions and all-date measured results."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent


def table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)


def main():
    p=json.loads((ROOT/'protocol.json').read_text());selected=json.loads((ROOT/'results/selected.json').read_text())
    meta=json.loads((ROOT/'figure_data.json').read_text());audit=json.loads((ROOT/'results/engine_verification.json').read_text())
    s=pd.read_csv(ROOT/'results/summary.csv');c=pd.read_csv(ROOT/'results/comparison.csv')
    control=pd.read_csv(ROOT/'results/comparison_to_previous.csv').query("subset=='all_240' and scale_name=='mean70'").iloc[0]
    mainrow=c.query("subset=='all_240' and scale_name=='mean70'").iloc[0]
    held=c.query("subset=='heldout_192' and scale_name=='mean70'").iloc[0]
    age=pd.read_csv(ROOT/'results/observation_age_daily.csv').query("scale_name=='mean70'")
    midnight=pd.read_csv(ROOT/'data/midnight_observations.csv')
    config=selected['mean70'];counts=meta['outcome_counts']
    order=['LYA','B32','B128','B512','STA','INC','DEC']
    frame=s.query("subset=='all_240' and scale_name=='mean70'").set_index('policy').loc[order].reset_index()
    settings=pd.DataFrame([dict(scale_name=k,alpha=v['alpha'],V=v['V']) for k,v in selected.items() if not k.startswith('_')])
    stale_fraction=100*age.epochs_over_300s.sum()/age.total_epochs.sum()
    text=r'''# Section 5.2：MLP mean 决策＋最近实测 PV（当前版本）

本版落实用户 2026-11-25 的两项要求：**决策端不访问噪声库；决策 PV 使用最近可观测功率乘以 MLP 平均时长。** 环境端继续产生 mean＋noise，并按实际 epoch 时间窗口积分得到实际 PV 能量。

## 1. 决策端与环境端

决策端先读取三个冻结 MLP 的均值：
\[
\hat m_T(s,a),\quad\hat m_{\mathrm{dem}}(s,a),\quad\hat m_\varepsilon(s,a).
\]

令 \(t_{\mathrm{obs}}=\max\{t_i:t_i\le t,\ P_i\text{ 有效}\}\)，则
\[
P_{\mathrm{last}}(t)=\alpha P_{\mathrm{raw}}(t_{\mathrm{obs}}),\qquad
\widehat E_{\mathrm{PV}}(s,a,t)=P_{\mathrm{last}}(t)\hat m_T(s,a),
\]
\[
\widehat E_s(s,a,t)=\left[\hat m_{\mathrm{dem}}(s,a)-P_{\mathrm{last}}(t)\hat m_T(s,a)\right]^+.
\]

LYA 选择
\[
a^*\in\arg\min_a\frac{V\widehat E_s+w_{\phi}Q\left(\hat m_\varepsilon-\sigma_e\hat m_T\right)}{\hat m_T}.
\]

这个短缺量是按用户要求形成的 **均值代入估计**，不再标成实际短缺的精确条件期望。它没有调用残差样本、未来 PV 曲线或积分函数。最近功率必须来自真实观测，不能使用可能借助下一条未来观测的 `power_at(t)` 插值结果。假设传感值在记录时间即可获取，不额外建模通信延迟。

动作确定后，环境从对应 action/epoch 的 30 个 donor 中联合抽取一个残差样本，生成实际 T、需求 D 和 ε。噪声沿用冻结模拟器的均值/物理边界校准，返回量满足 outcome=mean+noise。实际 PV 为
\[
E_{\mathrm{PV}}^{\mathrm{actual}}=\alpha\int_t^{t+T_{\mathrm{actual}}}P_{\mathrm{raw}}(u)\,du,
\qquad E_s^{\mathrm{actual}}=[D_{\mathrm{actual}}-E_{\mathrm{PV}}^{\mathrm{actual}}]^+.
\]

环境实际响应再更新 J、phase、Q 和时钟。LYA 使用的是历史响应形成的当前状态，不是把噪声从状态演化中去掉。每 task 重置训练状态，但 Q 和 PV 时钟跨 task 继续推进。

## 2. 决策隔离验证

- `controller.cpp` 的纯决策函数只接收三列 MLP 均值和 Q、V、阶段权重、σe、最近实测功率。参数中没有 epoch、donor、噪声库、时间轴或未来 PV 曲线。
- `controller.py` 提供独立 Python 决策接口，仅加载 `models.json` 与 `model_config.json`。在不提供任何噪声库文件的临时目录中完成 150 组状态验证，并与实验 C++ 结果一致。
- 改变未来 PV 的数值、删除未来 PV 支撑、改变环境噪声库，固定状态和已观测功率时评分与动作完全相同；改变噪声库仍会改变环境响应。
- 240 天真实观测查找、750 个模型状态、1,440 个 epoch 的独立 Python 轨迹回放均通过。回放中实际环境噪声不是零。
- 环境 PV 积分仍与公共 `PVDay.energy()` 一致，实际比较结果见 `results/engine_verification.json`。

## 3. 主结果：PV70、全部 240 天

![Figure 8](figures/figure8_pv70_mean_observed.png)

'''
    text+=f'''固定阶段权重 **(100,10,1)**，σe=0.24；只在原 48 个调参日期重新选择 V，PV70 最终仍选 **V={config['V']:g}**。

全 240 天平均 TARS：LYA **{mainrow.LYA_TARS_W:.6f} W**，最强 baseline B32 **{mainrow.baseline_TARS_W:.6f} W**，降低 **{mainrow.gain_percent:.4f}%**；95% 配对周 bootstrap 区间 **[{mainrow.gain_ci_low_percent:.4f}%, {mainrow.gain_ci_high_percent:.4f}%]**。当前七种策略中平均 TARS 排名为 {mainrow.mean_rank_all}。

LYA 的 7,680 次 PV70 运行观察到 **{mainrow.LYA_violations} 次** DWAE>0.24，最大 DWAE 为 **{mainrow.LYA_max_DWAE:.6f}**。相对每天最优可行 baseline，按每日均值比较 **{counts.get('win',0)} 胜、{counts.get('loss',0)} 负、{counts.get('tie',0)} 平**，不作逐日显著性声明。

{table(frame[['policy','mean_TARS_W','TARS_ci_low_W','TARS_ci_high_W','mean_DWAE','mean_hours','violations']])}

192 个未参与调参日期上，LYA 相对 B32 降低 **{held.gain_percent:.4f}%**，95% 区间 **[{held.gain_ci_low_percent:.4f}%, {held.gain_ci_high_percent:.4f}%]**。

旧信息模式下的递减权重 LYA 均值为 {control.previous_TARS_W:.6f} W；本版为 {control.current_TARS_W:.6f} W，相对旧版变化 **{control.current_TARS_W-control.previous_TARS_W:+.6f} W**。旧控制器读取噪声分布与未来积分 PV，不能将其结果混作本次受限信息策略的结果；两版仅作为历史对照报告。

## 4. PV 观测时效与缺口

仍使用原 240 天、同一 PV70 倍率 **α={config['alpha']:.15f}**、官方 `max_gap_s=None` 环境插值。48 个调参日上的平均供给/参考功率比为 70%，全部 240 天为 **{meta['all_dates_PV_percent']:.4f}%**。没有重新调整日期或倍率。

决策端在没有新观测时继续使用上一条有效实测值，并记录其时间戳和 age；没有把插值结果当新观测。240 个午夜中 **{int(midnight.age_s.gt(300).sum())} 个**的最新观测已超过 300 秒。最长发生在 **{midnight.loc[midnight.age_s.idxmax(),'date']}**，距午夜 **{midnight.age_s.max()/86400:.4f} 天**。这一极端值来自原记录的长中断，已保留在全量结果中。

PV70 的全部实际决策中，**{stale_fraction:.4f}%** 的观测 age 超过 300 秒，最大 age 为 **{age.max_observation_age_s.max()/86400:.4f} 天**。时间连续但观测不连续是这组数据的限制。环境跨缺口插值和控制器持有旧观测是两个不同处理步骤。

同一冻结参数的质量子集结果：

{table(c.query("scale_name=='mean70'")[['subset','days','LYA_TARS_W','gain_percent','LYA_violations']])}

## 5. 完整实验设置

- 260K simulator 全量版：600 条实测 profiling 观测拟合的三个 MLP 和经验联合噪声库；action=32/64/128/256/512，K=3，原 J/phase 定义及阈值保持冻结。
- 每日期从 00:00 开始，每策略 30 task×4 epoch。每日期每策略 32 个配对噪声种子 120000–120031。先对日内重复取均值，再对日期等权平均。
- 48 个原调参日期、8 个种子 110000–110007；仅搜索原 V 网格 10⁻⁴、10⁻³、10⁻²、10⁻¹、1、10、100，要求每次调参运行达标；不调 baseline 或权重形状。
- B32、B128、B512 固定 batch；STA=(32,32,512,512)，INC=(32,64,256,512)，DEC=(512,256,64,32)。六个 baseline 的原始记录完整复用，并经过回归核对。
- 新跑 **16,128 次调参＋46,080 次评估**；评估包括 6 个原 PV 倍率。加六个 baseline 和旧 LYA 控制，保存 368,640 条运行记录。主图仅使用 PV70 的七种当前策略，共 53,760 条运行。
- 2,000 次配对日历周 bootstrap；报告的区间不包含 MLP 拟合误差或长缺口内未知天气。重复模拟不是新 GPU 观测，动态 batch 切换仍需真实训练验证。

{table(settings)}

术语统一为 **LYA**、**TARS: time-averaged renewable shortfall**、**DWAE: duration-weighted average error**。图中算法不因权重或信息模式另造简称。

## 6. 调用与复现

独立决策接口（不需要噪声文件）：

```python
from controller import MeanOnlyLYA
policy = MeanOnlyLYA(model_directory, V=0.1, weights=(100,10,1), sigma_e=0.24)
decision = policy.select(J, phase, Q, observed_power_W)
action = decision['action']
```

`observed_power_W` 应由公共 PV observations 接口取最近历史样本并乘同一 α 后传入；独立控制器不读取 PV 数据。环境由 `runtime.py` 调用；`controller.cpp` 是完整实验使用的等价纯决策实现。

```bash
python verify_engine.py
python experiment.py prepare
python experiment.py tune
python experiment.py evaluate
python analyze.py
python prepare_figure_data.py
python plot_figure8.py
python plot_supplementary.py
python write_report.py
python finalize.py
```

主图：[PDF](figures/figure8_pv70_mean_observed.pdf)、[SVG](figures/figure8_pv70_mean_observed.svg)、PNG 与 600 dpi TIFF。所有逐日点和格对应值在 `source_data/`；全运行与配置在 `results/`、`protocol.json`；观测时效在 `data/midnight_observations.csv` 和 `results/observation_age_daily.csv`。

本目录是当前 Section 5.2 入口。历史 Figure 8 和 Section 5.3 结果仍使用此前的信息模式，不代表本版决策；本次不把历史结果重新命名为新结果。未改写冻结模型、原始 PV 数据库、历史实验或论文正文，也未自动同步 GitHub。
'''
    (ROOT/'README.md').write_text(text)
    notes=f'''TARS denotes time-averaged renewable shortfall and DWAE denotes duration-weighted average error. The 260K simulator executes 30 tasks of four epochs per policy from midnight, with 32 paired empirical-noise realizations on each of 240 dates. LYA uses decreasing phase weights (100,10,1), V={config['V']:g}, and a DWAE target of 0.24. Decisions use only the three MLP means and the latest valid PV measurement available by the decision time; estimated PV energy is this scaled power times mean epoch duration. Environmental outcomes use mean plus empirical joint noise and integrate actual PV over realized duration. (a) Gray intensity denotes full-day PV generation. Crosses indicate a smaller daily mean TARS for the best feasible baseline, circles indicate numerical ties, and unmarked cells indicate a smaller TARS for LYA. (b) Bars are equal-date mean TARS, with 95% intervals from 2,000 paired calendar-week bootstrap samples. Hatching identifies observed DWAE violations. (c) All 240 daily means per policy and seven overall means are shown; the dashed line denotes DWAE=0.24. The common PV multiplier α=4.19288 is calibrated on 48 tuning dates; 192 dates are excluded from tuning. Official unlimited-gap interpolation is used only by the environment, while the controller holds the latest actual measurement. Forty-five dates involve support gaps exceeding five minutes; observation-age and quality-subset results are reported separately.'''
    (ROOT/'FIGURE_CAPTIONS.md').write_text('# Figure 8. PV Generation and Algorithm Performance with Observed-Power Decisions.\n\n*Notes.* '+notes+'\n\nSupplementary figures use the same configurations at the five fixed capacity multipliers and the PV70 factor. Only V is tuned on the original tuning split. All daily observations are retained; numerical daily wins do not imply per-day significance.\n')
    texnotes=notes.replace('DWAE=0.24',r'$\mathrm{DWAE}=0.24$').replace('α=4.19288',r'$\alpha=4.19288$').replace('%',r'\%')
    (ROOT/'figure8_or.tex').write_text('\\begin{figure}[t]\n\\centering\n\\caption{PV Generation and Algorithm Performance with Observed-Power Decisions}\n\\label{fig:pv70-observed}\n\\includegraphics[width=\\textwidth]{figs/figure8_pv70_mean_observed.pdf}\n\\begin{minipage}{\\textwidth}\n\\footnotesize\\textit{Notes.} '+texnotes+'\n\\end{minipage}\n\\end{figure}\n')
    print('Saved mean-only decision report and captions.',flush=True)


if __name__=='__main__':main()
