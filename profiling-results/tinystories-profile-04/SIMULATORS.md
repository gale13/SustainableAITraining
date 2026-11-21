# Profiling 04：可行性实验与正式模拟器

按新的用途安排，同时保留以下两个目录。它们职责不同；全量校准版本不会覆盖独立留出实验。

| 目录 | 用途 | MLP 数据 | 噪声库 | 独立留出数据 |
| --- | --- | --- | --- | --- |
| [simulator](simulator/README.md) | 验证 simulator 可行性的重要实验，原样保留 | 360 条 train | 120 条 validation | 120 条 test |
| [simulator-full-data](simulator-full-data/README.md) | 实际使用的全量校准 simulator | 全部 600 条 | 全部 600 条 | 无 |

可行性版本对应 [GitHub commit 9e379438](https://github.com/gale13/SustainableAITraining/commit/9e379438019b951b1648fb27f28e3958c2b897f0)。全量版保存了可行性目录所有文件的 SHA-256；其 verify.py 会检查原目录是否保持不变。

全量版沿用已验证版本选出的三个网络配置、阶段阈值和经验残差抽样方式，重新训练全部模型参数，每个 action/epoch 使用 30 个 donor。旧 split 标签仅保留作来源追踪。评估泛化时引用可行性实验，运行模拟时使用 full-data 目录；全量版误差须称为“校准误差”，不能称为测试误差。
