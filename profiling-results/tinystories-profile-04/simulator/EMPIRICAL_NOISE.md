# 默认噪声：验证集实测残差的经验抽样

默认使用 `sim.step(action)`（等同于 `noise='empirical'` 或旧名称 `noise='independent'`）。每个 epoch 重新抽一个验证集 task，把该 task 在相同 batch、相同 epoch 下的三个残差一起使用。没有拟合正态分布，也没有用一个 sigma 替代噪声库。`noise='trajectory'` 仍可显式选用，作为连续四轮相关噪声的敏感性分析。

## 数据从哪里来

- Train：18 个配对种子，360 条 epoch 观测，训练三个 MLP。
- Validation：6 个配对种子，120 条观测，用于模型选择及建立噪声库。task 编号为 1、8、11、17、21、30。
- Test：另外 6 个配对种子，120 条观测，只用于检查拟合误差、噪声分布与 task 总量；不进入噪声库，不用来调残差尺度、尾部或抽样概率。

五个 action 为 32、64、128、256、512，每个 task 四轮。每个 `(action, epoch)` 格子有 6 个验证集残差三元组，等概率、有放回抽样。**逐轮独立的是 donor 抽取；同一轮的三个指标成组抽取，不能拆成三个独立噪声。**

默认方式由用户选择为经验残差抽样，并以逐轮新抽样保持模拟器的条件均值；不是根据测试集上哪条曲线更好临时选择。已有报告中的 `independent` 数值结果就是这个默认方式的结果。

## 实际抽取的残差及均值校准

三个输出记为 $Y=(T,D,H)$，其中 $D$ 是 GPU demand，单位 J；$H=T J_{next}$，单位 s，对应第三个 m 函数的标签。MLP 在给定状态和动作下输出 $\hat m=(\hat T,\hat D,\hat H)$。

为适应不同状态的均值大小并满足物理范围，库中保留相对残差，而非直接把原始单位的误差平移到任意状态。每条验证观测 $i$ 存储：

$$
u_{T,i}=T_i/\hat T_i,\qquad
u_{D,i}=D_i/\hat D_i,\qquad
v_i=J_{next,i}/J_i-\hat H_i/(\hat T_i J_i).
$$

前两项是“1 + 相对残差”，第三项是下一轮 J 保持比例的残差。它们来自同一条实测记录，三元组不拆散。

在当前 `(a, epoch)` 的六条候选中，令 $w_{T,i}=u_{T,i}/\overline u_T$、$w_{D,i}=u_{D,i}/\overline u_D$，以及 $q=\hat H/(\hat T J)$。取一个共同平移量 $c$，使

$$
\frac16\sum_{i=1}^{6}w_{T,i}\,\operatorname{clip}(q+v_i+c,10^{-12},1)=q.
$$

再均匀抽取 $I\in\{1,\ldots,6\}$，得到

$$
T=\hat T w_{T,I},\qquad D=\hat D w_{D,I},\qquad
J_{next}=J\operatorname{clip}(q+v_I+c,10^{-12},1),\qquad H=TJ_{next}.
$$

这仍是有限实测样本的经验分布，没有添加高斯随机数。尺度和共同平移量只用已训练模型、当前输入和验证噪声库计算。校准使 $E[T]=\hat T$、$E[D]=\hat D$、$E[H]=\hat H$，同时保证时长、能耗为正，$0<J_{next}\le J$。边界裁剪可能改变 loss 扰动尾部，因此不能声称直接、原样复制了所有原始残差的分布。

可以统一写为 **mean + noise**：

$$
Y_r=\hat m(s_r,a_r)+\xi_r,\qquad
\xi_r=G(\hat m(s_r,a_r),J_r,B_{a_r,r,I_r})-\hat m(s_r,a_r).
$$

在固定校准库的模拟器内，默认模式每轮重新、独立抽取 $I_r$，因此 $E[\xi_r\mid\text{历史},a_r]=0$（数值精度内）。噪声分布可以随 action、epoch 变化；**这不意味着实测数据支持 action-independent 的同分布加性噪声，也不证明真实训练的状态充分性。**

## 使用与核查

```python
from simulator import Simulator

sim = Simulator(seed=42)
rows = [sim.step(128) for _ in range(4)]
for row in rows:
    print(row['T_s'], row['mean_T_s'], row['noise_T_s'],
          row['noise_source'], row['noise_donor_task'])
```

每行输出三个实际值、`mean_*` 和 `noise_*`，满足“实际值 = mean + noise”。`noise_source` 为 `validation`，并记录 `noise_donor_task/action/epoch`。`python verify.py` 检查噪声库来源、默认抽样、均值校准和物理范围。

这套经验分布保留验证样本已有的偏斜、离散性及指标间关联，不外推未见过的极端事件。每格仅 6 个 donor，且验证集也参与 MLP 选择，因此不应把有限样本尾部当成已准确估计的真实尾部。完整测试诊断和区间覆盖率见 [NOISE_REPORT.md](NOISE_REPORT.md)。
