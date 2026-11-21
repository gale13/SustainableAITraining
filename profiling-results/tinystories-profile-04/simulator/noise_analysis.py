"""Noise report and exact four-epoch empirical rollout distributions.

The test set is used only for diagnostics, never to change the noise bank.
Both independent (6**4 paths) and trajectory (6 paths) distributions are
enumerated, eliminating Monte Carlo variation from their comparison.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mpl-profile04-refit')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/profile04-refit-cache')
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import ROOT, ACTIONS, METRICS, Simulator, phase_after
from compare import table, export


def enumerate_paths(sim,action,mode):
    sim.reset()
    output=[]
    def visit(j,phase,epoch,t,d,donor):
        if epoch==4:
            output.append((t,d,j));return
        cases=sim.scenarios(j,phase,action,epoch,'independent')
        indices=range(len(cases)) if mode=='independent' or epoch==0 else [donor]
        for index in indices:
            ti,di,_,jp=cases[index]
            visit(jp,phase_after(phase,j,jp,sim.config),epoch+1,t+ti,d+di,index if epoch==0 else donor)
    visit(sim.J,1,0,0.,0.,None)
    assert len(output)==(6**4 if mode=='independent' else 6)
    return np.asarray(output)


def crps(sample,truth):
    x=np.sort(sample);n=len(x)
    half_pairwise=np.dot(2*np.arange(n)-n+1,x)/n**2
    return np.mean(np.abs(x[:,None]-truth[None,:]))-half_pairwise


def main():
    data=pd.read_csv(ROOT/'epochs.csv');test=data[data.split=='test']
    noise=pd.read_csv(ROOT/'noise_summary.csv');pooled=noise[noise.group=='all'].copy()
    tests=pd.read_csv(ROOT/'action_tests.csv')
    common=pd.read_csv(ROOT/'common_component.csv')
    serial=pd.read_csv(ROOT/'serial_correlation.csv')
    coverage=pd.read_csv(ROOT/'interval_coverage.csv')
    oldnoise_path=ROOT/'comparison_reference/noise_summary.csv'
    if oldnoise_path.exists():
        old=pd.read_csv(oldnoise_path);old=old[old.group=='all']
        difference=old[['metric','mean','sd','skew','excess_kurtosis']].merge(
            pooled[['metric','mean','sd','skew','excess_kurtosis']],on='metric',suffixes=('_old','_new'))
        difference.to_csv(ROOT/'noise_comparison.csv',index=False)
    # Resample whole test seeds for lag intervals, preserving all within-task pairs.
    rng=np.random.default_rng(20261121)
    for row in serial.itertuples():
        x=test[test.action==row.action].pivot(index='task',columns='epoch',values='res_'+row.metric).to_numpy()
        estimates=[]
        for _ in range(2000):
            xb=x[rng.integers(0,len(x),len(x))]
            estimates.append(np.corrcoef(xb[:,:-1].ravel(),xb[:,1:].ravel())[0,1])
        serial.loc[row.Index,'seed_bootstrap_low']=np.nanquantile(estimates,.025)
        serial.loc[row.Index,'seed_bootstrap_high']=np.nanquantile(estimates,.975)
    serial.to_csv(ROOT/'serial_correlation.csv',index=False)
    actual=test.groupby(['task','action'],as_index=False).agg(total_T=('T','sum'),total_dem=('dem','sum'))
    actual.to_csv(ROOT/'observed_test_task_totals.csv',index=False)
    sim=Simulator(seed=20261121);samples=[];scores=[]
    for action in ACTIONS:
        truth=actual[actual.action==action]
        for mode in ['independent','trajectory']:
            values=enumerate_paths(sim,action,mode)
            for i,(t,d,j) in enumerate(values):
                samples.append(dict(action=action,mode=mode,path=i,probability=1/len(values),total_T=t,total_dem=d,J_end=j))
            for col,index in [('total_T',0),('total_dem',1)]:
                x=values[:,index];y=truth[col].to_numpy();lo,hi=np.quantile(x,[.05,.95])
                scores.append(dict(action=action,mode=mode,metric=col,n_paths=len(x),n_test_tasks=len(y),
                    observed_mean=y.mean(),predicted_mean=x.mean(),observed_sd=y.std(ddof=1),
                    predicted_sd=x.std(ddof=0),mean_error=x.mean()-y.mean(),
                    wasserstein=stats.wasserstein_distance(x,y),crps=crps(x,y),q05=lo,q95=hi,
                    coverage90=np.mean((y>=lo)&(y<=hi))))
        print('Exact task distributions complete:',action,flush=True)
    sample=pd.DataFrame(samples);score=pd.DataFrame(scores)
    sample.to_csv(ROOT/'exact_task_distributions.csv',index=False)
    score.to_csv(ROOT/'task_distribution_scores.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,5,figsize=(16,7),layout='constrained')
    for col,action in enumerate(ACTIONS):
        for row,(metric,scale,label) in enumerate([('total_T',60,'Four-epoch total training duration (min)'),
                                                 ('total_dem',1000,'Four-epoch total GPU demand (kJ)')]):
            ax=axs[row,col]
            for mode,color,title in [('independent','#4477AA','Independent epochs'),('trajectory','#CC6677','Joint four-epoch donor')]:
                x=np.sort(sample[(sample.action==action)&(sample['mode']==mode)][metric].to_numpy()/scale)
                ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=title,color=color,lw=1.4)
            y=np.sort(actual[actual.action==action][metric].to_numpy()/scale)
            ax.step(y,np.arange(1,len(y)+1)/len(y),where='post',label='Real test tasks (6)',color='#222222',lw=1.2)
            ax.scatter(y,np.arange(1,len(y)+1)/len(y),color='#222222',s=15,zorder=4)
            ax.set(title=f'Batch {action}',xlabel=label,ylabel='Cumulative probability',ylim=(0,1.04))
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    axs[0,0].legend(fontsize=7,loc='lower right')
    fig.suptitle('Exact simulator distributions versus held-out tasks\n'
                 'Independent: 1,296 paths per batch. Joint: 6 paths per batch. All noise donors are validation seeds.',fontsize=12)
    export(fig,'task_total_noise_comparison')
    units={'T':'s','dem':'J','epsilon':'s'}
    pooled.insert(1,'unit',pooled.metric.map(units))
    sds=noise[noise.group!='all'].pivot(index='group',columns='metric',values='sd')
    sds.index=sds.index.astype(int);sds=sds.sort_index().reset_index().rename(columns={'group':'batch'})
    wins={}
    for metric in ['total_T','total_dem']:
        paired=score[score.metric==metric].pivot(index='action',columns='mode',values='crps')
        wins[metric]={'joint_lower_crps_batches':int((paired.trajectory<paired.independent).sum()),
                      'independent_lower_crps_batches':int((paired.independent<paired.trajectory).sum())}
    overview=score.groupby(['metric','mode'])[['crps','wasserstein','coverage90']].mean().reset_index()
    (ROOT/'noise_analysis_summary.json').write_text(json.dumps({'comparison':wins,
        'normal_noise_supported':False,'single_action_independent_additive_noise_supported':False,
        'test_seeds':6,'test_epoch_observations':120,'noise_bank_source':'validation only',
        'independent_paths_per_batch':1296,'trajectory_paths_per_batch':6,
        'test_results_do_not_change_default_sampler':True,
        'default_sampler':'empirical',
        'default_alias':'independent',
        'sampling':'fresh uniform validation donor per epoch; three metrics jointly; condition on action and epoch',
        'distribution_family':'nonparametric empirical residuals; no Gaussian assumption',
        'default_selection':'user-selected empirical method; fresh draws preserve conditional means; no test tuning',
        'validation_tasks':[1, 8, 11, 17, 21, 30],
        'validation_epoch_observations':120,
        'donors_per_action_epoch':6},indent=2)+'\n')
    report=f'''# 新版 Profiling 04：残差分布、动作差异与四轮联合噪声

本报告分析 online `train_loss` 状态版本；权重、阈值和 train/validation/test 划分均不因本报告而改变。对三个条件均值定义残差 eta_X = X - m_X(J,phi,a)。全部 120 个 test 点来自 6 个留出的配对 seed、30 条四轮轨迹。没有去除异常值或截断尾部。

## 主要判断

三个目标的总体残差均明显偏离一个共同加性正态分布，不能用一个已经验证的 Gaussian sigma 描述。下表 SD 是经验标准差。它包含外部波动、模型拟合偏差、初始种子影响以及压缩状态未覆盖的训练历史，不能直接称为纯外生随机性。

{table(pooled,['metric','unit','mean','sd','sd_ci_low','sd_ci_high','q05','q95','skew','excess_kurtosis','shapiro_p'])}

SD 区间按整 seed 重采样 2,000 次，保留同 seed 的全部 action/epoch 配对。直方图覆盖所有点；橙线是匹配经验均值与 SD 的正态密度，不是拟合已获验证的噪声模型。Q–Q 图对比相同正态的分位数。

![残差直方图与正态Q–Q对比](noise.png)

时长和时间加权 loss 右偏、厚尾；需求残差也有偏度和超额峰度。样本存在跨轮及配对依赖，Shapiro–Wilk 的 iid 前提不成立，因此其 p 值只作形状诊断，不能当作 120 次独立重复的严格显著性结论。三个总体检验的 Holm 校正数值也存入 noise_summary.csv。

## Action-independent 假设能否成立

各 batch 的原单位残差 SD：

{table(sds)}

单位：T、epsilon 为秒，dem 为焦耳。原单位下，不同 batch 的分布/尺度明显不同。不能从所有 batch 的残差混合直方图提取一个 sigma，再假设该噪声对任何动作都适用。

![按batch的全部残差](noise_by_action.png)

只用 validation 为每个 action 估计残差均值、SD，再把 test 残差标准化。对 action 分布差异，按 seed 内交换完整四轮轨迹的 action 标签，共 4,999 次；统计量是七个共同分位点处的经验 CDF 差异。

{table(tests,['metric','scope','p','p_holm_three'])}

raw 是原单位残差，initial_state_only 只看同一初始状态的第一轮，validation_standardized 是验证集标准化残差。新 epsilon 在标准化后仍显示动作间差异。T、dem 的部分校正结果没有超过常用阈值，这只是证据不足，不是 action independence 已成立。动作固定对应 GPU，实际执行时刻不同且没有随机交换 GPU；置换分数只能描述这些配对轨迹的差异，不能证明动作的因果效应。跨 epoch 的状态分布也会改变混合残差的形状。

![验证集标准化后的残差](standardized_noise.png)

## 能否提取一个真实的共同外生噪声

作描述性探索：对各 action 的 test 残差分别去均值，再把同 seed、同 epoch 的五个残差取均值。该均值只是共同成分的代理。它的方差包含各 action 独有噪声和拟合误差，不能直接当作共同外因的方差。进一步用另外四个 action 的平均残差预测当前 action，评估 leave-one-action-out skill（相对零预测）：

{table(common,['metric','sd','mean_offdiagonal_covariance','loo_corr','loo_skill_vs_zero'])}

三个 skill 都为负，未得到一个能够改善预测的共同加性成分。同 seed/epoch 也不意味着相同墙钟时刻，不足以识别同步外部扰动。公共模拟器不将这个描述性代理回填噪声库。

## 跨轮、跨目标相关性

按 action 计算相邻轮残差相关系数，每项只有 6 条轨迹、18 对相邻值；下表区间按整 seed bootstrap 2,000 次，样本很少，不能将高相关当成稳定规律。跨目标 Pearson 相关另见 residual_correlations.csv。

{table(serial,['metric','action','lag1_corr','seed_bootstrap_low','seed_bootstrap_high'])}

## 模拟器中的可操作噪声表示

采用验证集实测残差的经验分布，只用 6 个 validation seed 的 120 条观测构成噪声库，不拟合参数分布。完整公式、mean + noise 表示及调用方式见 [EMPIRICAL_NOISE.md](EMPIRICAL_NOISE.md)。每条记录同时保存 T 的比例残差、dem 的比例残差、J_next/J 的保留率残差。三项一起抽取，保留本 epoch 的耦合关系；在同一状态下校准有限场景，使 E[T]、E[dem]、E[T*J_next] 恢复为三个 MLP 均值，并保证正时长、正需求、0<J_next≤J。

- **empirical（默认；independent 为兼容名称）**：每个 epoch 独立抽一个 validation seed 的对应 action/epoch 三元残差。下方图表中的 independent 就是该默认方式。
- trajectory（显式选用）：task 开始时抽一个 validation seed，连续四轮使用该 seed 的对应残差，用于相关噪声敏感性分析。
- pooled：混合 action 的对照方式，用于诊断；不作为 action-independent 已验证的证据。

**条件均值的边界：** 固定输入状态下，有限场景库的等权平均与 MLP 均值一致。逐轮独立抽样按这个分布抽取；但 trajectory 固定同一 donor 后，donor 会与已观测历史和当前状态相关，其后验分布不再必然均匀。因此不能把场景均值检查当作论文“给定完整历史仍只依赖 s,a”的证明。联合轨迹模式适合检验相关噪声下的表现；当前默认的 empirical/independent 在模拟器内满足该历史条件均值性质；这不构成真实数据的状态充分性证明。

随机 donor 索引的抽取概率可以与 action 无关，但从索引到实际扰动的映射仍依赖 action 和 epoch。因此这是“共同随机输入 + 动作相关响应”的经验表示，不能改写成已识别的同分布加性外生噪声。用户选择了经验残差方法；默认改为逐轮重新抽样，以保留 mean + noise 的条件均值解释，没有根据 test 结果挑选默认模式。测试数据不用于更新噪声库或校准。

## 单轮区间及四轮总量对比

条件场景的经验 5%–95% 区间在 test 上的覆盖率：

{table(coverage)}

这些区间来源于只有 6 个 validation donor 的有限经验分布，不是有保证的预测区间。尾部不足、模型偏差或新环境会导致明显欠覆盖；不能宣称 90% 区间已经可靠校准。

为比较连续抽样是否更接近真实 task，总时长和总需求均累计全部四轮。independent 枚举全部 6^4=1,296 条路径，trajectory 枚举全部 6 条路径；每条路径递推 J、phi。两种方式都用 train 初始 loss 均值开始（J(0)≈1），不用 test 的未来数据。这样比较没有蒙特卡洛抽样波动。

![四轮总时长与能耗的经验分布](task_total_noise_comparison.png)

下表按五个 batch 等权平均；CRPS 和 Wasserstein 越小越好，单位与总量一致（total_T 为秒，total_dem 为焦耳），coverage90 越接近 0.90 越好。

{table(overview)}

按逐 batch CRPS，联合抽样在时长上更接近实测的 batch 数为 {wins['total_T']['joint_lower_crps_batches']}/5，在能耗上为 {wins['total_dem']['joint_lower_crps_batches']}/5。这里只描述这 6 个 test seed 的结果，不据此调参、修改噪声库或宣称普遍优于独立抽样。逐 batch 全指标见 task_distribution_scores.csv，所有有限路径及概率见 exact_task_distributions.csv，真实测试总量见 observed_test_task_totals.csv。

## 复现与文件

```bash
python diagnostics.py
python plots.py
python noise_analysis.py
python verify.py
```

`diagnostics.py` 计算残差、分布差异和基础模拟器检查；`plots.py` 绘制全部观测；`noise_analysis.py` 计算整 seed 的跨轮相关区间、枚举四轮分布并生成本报告。参数权重和验证集噪声库均保持重拟合版本。`noise_comparison.csv` 比较旧 probe 与新 online loss 的残差统计（若打包了旧统计）。

方法参考：[SciPy Shapiro–Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)、[SciPy Wasserstein distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)。这些是计算方法，不替代本报告对依赖观测、动作/GPU混杂和小样本的限制说明。
'''
    (ROOT/'NOISE_REPORT.md').write_text(report)
    print(overview.to_string(index=False));print(json.dumps(wins))

if __name__=='__main__':main()
