"""Generate full-calibration figures and documentation; never label them test scores."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mpl-profile04-full-data')
os.environ.setdefault('XDG_CACHE_HOME','/private/tmp/profile04-full-data-cache')
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulator import ROOT,ACTIONS,METRICS,Simulator

def table(frame):
    def fmt(x):return f'{x:.8g}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def export(fig,name):
    for extension in ['png','pdf','svg']:
        fig.savefig(ROOT/(name+'.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    data=pd.read_csv(ROOT/'epochs.csv');scores=pd.read_csv(ROOT/'metrics.csv')
    models=json.loads((ROOT/'models.json').read_text());cfg=json.loads((ROOT/'model_config.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.16})
    colors=['#4477AA','#EE6677','#228833','#CCBB44','#AA3377']
    labels={'T':('Epoch duration','s'),'dem':('GPU energy demand','J'),'epsilon':('Time-weighted loss $T J_{next}$','s')}
    axis_names={'T':'duration T','dem':'GPU energy demand','epsilon':'time-weighted loss $T J_{next}$'}
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.6),layout='constrained')
    for ax,key in zip(axes,METRICS):
        name,unit=labels[key]
        for action,color in zip(ACTIONS,colors):
            group=data[data.action==action]
            ax.scatter(group[key],group['pred_'+key],s=10,alpha=.42,color=color,label=f'Batch {action}',edgecolors='none')
        values=data[[key,'pred_'+key]].to_numpy();lo,hi=values.min(),values.max();pad=(hi-lo)*.04
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#555555',lw=.9,ls='--')
        ax.set(xlabel=f'Observed {axis_names[key]} ({unit})',ylabel=f'MLP predicted {axis_names[key]} ({unit})',
               title=f'{name}\nCalibration RMSE = {scores.set_index("metric").loc[key,"rmse"]:.4g} {unit}')
        if key=='dem':ax.ticklabel_format(axis='both',style='sci',scilimits=(0,0),useMathText=True)
    axes[0].legend(fontsize=7,loc='upper left')
    fig.suptitle('All 600 calibration observations; no held-out observations in this version',fontsize=11)
    export(fig,'calibration_fit')
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.4),layout='constrained')
    for ax,key in zip(axes,METRICS):
        name,unit=labels[key];res=data['res_'+key]
        ax.hist(res,bins=35,color='#4477AA',alpha=.8,edgecolor='white',linewidth=.35)
        ax.axvline(0,color='#333333',lw=.8)
        ax.set(xlabel=f'Residual of {axis_names[key]} ({unit})',ylabel='Number of calibration observations',title=name)
    fig.suptitle('600 in-sample residuals (observed minus fitted); pooled descriptively, no Gaussian fit',fontsize=11)
    export(fig,'residual_histograms')
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.4),layout='constrained')
    for ax,key in zip(axes,METRICS):
        name,unit=labels[key]
        for action,color in zip(ACTIONS,colors):
            values=np.sort(data.loc[data.action==action,'res_'+key].to_numpy())
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,label=f'Batch {action}',lw=1.15)
        ax.set(xlabel=f'Residual of {axis_names[key]} ({unit})',ylabel='Empirical cumulative probability',title=name,ylim=(0,1.02))
    axes[0].legend(fontsize=7)
    fig.suptitle('Action-specific residual distributions: 120 calibration observations per action',fontsize=11)
    export(fig,'residual_ecdf_by_action')
    residuals=[]
    for key in METRICS:
        for group,frame in [('all',data)]+[(str(a),data[data.action==a]) for a in ACTIONS]:
            x=frame['res_'+key]
            residuals.append(dict(metric=key,action_group=group,n=len(x),mean=x.mean(),sd=x.std(ddof=1),
                q01=x.quantile(.01),q05=x.quantile(.05),median=x.median(),q95=x.quantile(.95),q99=x.quantile(.99),
                skew=x.skew(),excess_kurtosis=x.kurt()))
    pd.DataFrame(residuals).to_csv(ROOT/'noise_summary.csv',index=False)
    selected=pd.DataFrame([dict(metric=key,width=m['width'],parameters=m['parameters'],L2=m['penalty'],
        initialization_seed=m['initialization_seed'],iterations=m['iterations'],converged=m['converged']) for key,m in models.items()])
    selected.to_csv(ROOT/'selected_models.csv',index=False)
    example=[]
    for action in ACTIONS:
        sim=Simulator(seed=42)
        example.extend(sim.step(action) for _ in range(4))
    pd.DataFrame(example).to_csv(ROOT/'example_rollouts.csv',index=False)
    display=scores[['metric','n','mse','rmse','mae','r2']].copy()
    report=f'''# Profiling 04 正式使用版：全量数据校准

本目录使用全部 **600 条 epoch 观测** 重新训练三个 MLP，并使用全部观测构建经验残差库。共 30 个配对随机种子 × 5 个 batch × 4 个 epoch，即 150 次训练 task。它用于实际运行模拟；独立留出验证的可行性实验证据保留在同级 `simulator/`，对应 GitHub commit `9e379438019b951b1648fb27f28e3958c2b897f0`。该实验的全部文件校验值保存在 feasibility_snapshot_sha256.json，verify.py 已核对其未改变。

## 两个版本的职责

| 项目 | 保留的可行性实验 | 本目录：全量校准模拟器 |
| --- | --- | --- |
| MLP 拟合数据 | 原 train：360 条 | 全部 600 条 |
| 配置选择 | validation：120 条 | 固定已选配置，不重新搜索 |
| 噪声库 | validation：120 条 | 全部 600 条拟合后残差 |
| 每个 action/epoch 的 donor 数 | 6 | 30 |
| 独立 test | 120 条 | 无 |
| 误差含义 | 包含留出测试误差 | 仅全量校准误差 |

epochs.csv 中 original_split 保留原来的 train/validation/test 身份以便追踪；该列不用于任何筛选。三类原始观测均进入模型拟合和噪声库。calibration_role 对全部记录均为 full_data。

## 状态、模型及训练设置

状态仍为 s=(J,phi)，action 为 batch size，支持 32、64、128、256、512。J 是已完成 online train_loss 的任务内 running minimum，除以训练前实测初始 probe loss 加 1e-8；当前 epoch 的结果不会进入该 epoch 的决策输入。K=3、W=1，阶段阈值固定为 {cfg['beta_internal']}，完全沿用可行性实验。初始 J 约为 1，默认初始 loss 使用全部 150 次 task 的初始实测值均值。也可传入实际初始 loss。

三个目标分别为 T（单轮训练时长，秒）、dem（GPU 训练窗口的需求能量，焦耳，不含 pv）、epsilon=T*J_next（秒）。输入仍为标准化 J、phase one-hot、action one-hot，共 9 维；J 及目标尺度在全量数据上重新计算。每个 MLP 一个 tanh 隐层，T、dem 使用正值输出；epsilon 输出限定在 (0,J*predicted_T)，直接拟合 T*J_next。

{table(selected)}

网络宽度和 L2 沿用原验证集选定值；所有权重从头拟合。每个网络使用 20261119、20261120 两次初始化，按全量数据的正则化目标选较低者。优化为 L-BFGS-B，最多 6,000 次迭代、ftol=1e-11、gtol=1e-7；所有最终选中网络均收敛。epsilon 拟合使用已训练、冻结的 T 网络作为输出上界。详细记录见 optimization_runs.csv。

## 全量校准的最终误差

**以下所有指标均在参与拟合的 600 条数据上计算，不是 test loss，不能用来重新证明泛化效果。**

{table(display)}

T、epsilon 的 RMSE/MAE 单位为秒，dem 为焦耳；对应 MSE 单位为秒²或焦耳²。metrics.csv 同时提供标准化 MSE 与含 L2 的优化目标，两者不可混用。分 action 指标见 metrics_by_action.csv。总体 R² 也包含不同 batch 的均值差异，不代表轮间微小波动已被准确预测。

![全部600条校准观测与MLP输出](calibration_fit.png)

## 噪声：全量经验残差抽样

noise_bank.npz 的形状是 **30 × 5 × 4 × 3**，分别对应配对 task 编号、action、epoch、三个共同抽取的残差成分。noise_library.csv 列出全部 600 个条目的实测值、预测、原单位残差、实际保存的比例残差及原始 split，便于逐条核查。

默认每轮均匀、有放回地从对应 action/epoch 的 30 个 donor 中重新抽一个，三个指标一起抽。没有拟合正态分布，也不通过一个 sigma 替代残差库。与可行性实验相同，内部采用相对残差和均值/物理边界校准；完整公式见 [EMPIRICAL_NOISE.md](EMPIRICAL_NOISE.md)。返回结果含 mean、noise 和 donor 来源，满足 outcome=mean+noise。noise_source 明确为 full_data_calibration。

![全量拟合残差直方图](residual_histograms.png)

![按动作划分的经验残差分布](residual_ecdf_by_action.png)

noise_summary.csv 给出描述性均值、经验 SD、分位数、偏度和超额峰度。直方图合并所有 action 仅为展示，真正抽样仍按 action/epoch 分组，未删除任何异常值。

**本版本的噪声库来自参与 MLP 拟合的数据，因此误差幅度可能偏乐观。** 600 条观测来自 30 个种子，不是 600 次相互独立实验。更多 donor 可以覆盖更多已观测情况，不能补出未观测极端尾部；可行性验证结论仍以保留版本为准。固定 batch、固定 GPU 的原始实验也未验证任意 batch 切换后的真实训练动力学。

## 直接使用与重建

在本目录运行，推理只依赖 NumPy：

```python
from simulator import Simulator
sim = Simulator(seed=42)
rows = [sim.step(128) for _ in range(4)]
total_seconds = sum(r['T_s'] for r in rows)
total_energy_J = sum(r['demand_J'] for r in rows)
print(total_seconds, total_energy_J)
```

每次 task 四轮后调用 sim.reset()；默认每轮重新抽样。noise='trajectory' 可显式选用同一 donor 连续四轮的相关噪声敏感性模式；noise='none' 返回三个 MLP 均值。example_rollouts.csv 提供五种固定 action 各一次四轮模拟，仅为运行示例。

重建命令（本地 CPU 即可，不会启动原始 GPU 训练）：

```bash
python -m pip install -r requirements.txt
python fit.py
python verify.py
python report.py
```

verify.py 检查全部数据参与拟合、600 条噪声条目重构、状态因果顺序、所有 600 个实测状态的有限场景均值和物理边界、随机种子可复现以及可行性实验文件未变。结果见 verification.json。source.zip 是原始观测快照；calibration_recipe.json 固定了本次配置；checksums.json 记录本目录文件哈希。
'''
    (ROOT/'README.md').write_text(report)
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts and p.name!='checksums.json'}
    (ROOT/'checksums.json').write_text(json.dumps(files,indent=2)+'\n')
    print('Wrote full-calibration report, 3 figures in PNG/PDF/SVG, noise summaries and sample rollouts.')

if __name__=='__main__':main()
