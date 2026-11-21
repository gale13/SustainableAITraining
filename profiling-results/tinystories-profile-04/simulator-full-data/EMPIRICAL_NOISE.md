# 全量校准的经验噪声

本版本不设置独立的 validation/test，原有三种身份的全部 600 条观测均用于重新拟合 MLP 和构建噪声库。原留出实验单独保留。每个 action、epoch 有 30 个 donor，对应 30 个不同配对种子；每轮重新等概率抽取一个 donor，同一轮的三个指标一起取出。

记输出 Y=(T,D,H)，H=T J_next，三个 MLP 的均值预测为 (mT,mD,mH)。每条校准记录 i 保存：

```
uT_i = observed_T_i / predicted_T_i
uD_i = observed_D_i / predicted_D_i
v_i  = observed_J_next_i / J_i - predicted_H_i / (predicted_T_i * J_i)
```

uT、uD 是“1+相对残差”，v 是下一轮 J 保持比例的残差，三项来自同一实测记录。它们保留测量所呈现的分布特征，不强行拟合 Gaussian。时长、能耗的原单位残差也完整保存在 noise_library.csv，以便审计。

对当前 action、epoch 的 30 个候选，令 wT_i=uT_i/mean(uT)，wD_i=uD_i/mean(uD)，q=mH/(mT*J)。通过二分求共同平移量 c，使：

```
mean_i(wT_i * clip(q + v_i + c, 1e-12, 1)) = q
```

等概率抽 I，产生：

```
T      = mT * wT_I
D      = mD * wD_I
J_next = J * clip(q + v_I + c, 1e-12, 1)
H      = T * J_next
noise  = (T,D,H) - (mT,mD,mH)
```

在已校准模拟器内，这使候选场景均值与三个 MLP 输出一致，并保证 T,D>0、0<J_next≤J、H=T*J_next。相对尺度和边界处理意味着这不是把原单位残差不加修改地平移到任意状态；loss 尾部也可能受边界影响。

默认每轮重新抽 I，因此在该模拟器内给定已发生历史、当前动作，noise 的条件均值为零。索引 I 的均匀抽样与 action 无关，但其对应残差及尺度依赖 action、epoch；这不等同于数据已支持一个共同的 action-independent 加性噪声分布。

`noise='trajectory'` 在 task 开始时取一个 donor，四轮固定使用同一 donor。它保留跨轮残差关联，用于敏感性分析；donor 与后续历史相关，不能据均匀场景均值直接声称其历史条件均值仍等于 MLP 输出。

所有 residual 均为最终全量 MLP 的 **in-sample residual**，并非交叉拟合误差或新的独立测试误差。原始 mean 拟合偏差、未入状态的训练历史和种子差异也会进入残差。因此经验 SD 是这套校准数据的描述性量，不是已识别的纯外生 sigma。相关独立评估以同级保留的可行性实验为准。
