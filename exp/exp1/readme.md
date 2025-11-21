
实验1：评估AgE-ASHA两阶段正则化选择算法的有效性实验

与现有的三种最优的超参数优化算法 ASHA，Hyperband、BOHB 进行对比。探索相同规模 N=2000的正则化配置对模型进行训练，报告模型的测试精度和时间消耗
数据集 Adult，Ccfraud，Clickpred, Connect, Dionis，Walking，Frappe，Diabetes，Criteo
模型结构：MLP(512*6)
对比基线：ASHA、Hyperband、BOHB
评价指标：balanced Acc & Spend time
Age-ASHA 探索阶段的数据采样率设置为 0.2，利用阶段的探索利用率设置为 1/5，即利用 400 个


batch_size = 64
seed = 42
lr = 1e-3
max_epochs = 4
eta = 2
num_samples = 2000
k_n = 0.2
population_size = 10
sample_size = 3
max_steps = 300
sample_ratio = 0.2
swa_start_epoch = 1
crossover_rate = 0.1
mutation_rate = 0.6
random_rate = 0.3


-----------------------------------------------------------------------------

数据集 Adult : 收入预测数据集
Batch size 64，探索阶段最大 300 batch, 利用阶段最大 4 epochs，eta=2

Baseline            Balanced Acc    Time usage      Speed up
MLP                 0.818698        -               -
ASHA                0.829461        4755.27s        3.08 x
HyperBand           0.829658        8002.59s        5.18 x
BOHB                0.830157        8966.41s        5.80 x
AgE-ASHA(ours)      0.828993        1546.01s        1.00 x

-----------------------------------------------------------------------------

数据集 Ccfraud : 欺诈检测数据集
Batch size 128，探索阶段最大 300 batch, 利用阶段最大 4 epochs，eta=2

Baseline            Balanced Acc    Time usage      Speed up
MLP                 0.951145        -               -
ASHA                0.958702        10221.26s        3.86 x
HyperBand           0.959238        16275.46s        6.14 x
BOHB                0.957691        16163.59s        6.10 x
AgE-ASHA(ours)      0.958166        2649.83s         1.00 x

-----------------------------------------------------------------------------

数据集 Connect : 游戏结果预测数据集
Batch size 64，探索阶段最大 300 batch, 利用阶段最大 4 epochs，eta=2


Baseline            Balanced Acc    Time usage      Speed up
MLP                 0.694853        -               -
ASHA                0.750893        6407.78s        3.85 x
HyperBand           0.753304        10044.81s       6.04 x
BOHB                0.748334        9583.75s        5.76 x
AgE-ASHA(ours)      0.752686        1663.35s        1.00 x

-----------------------------------------------------------------------------

