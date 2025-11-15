
实验1：评估AgE-ASHA两阶段正则化选择算法的有效性
评估相同数量的参数配置，分析平衡准确率和使用时间
数据集 Adult，Ccfraud，Clickpred, Connect, Dionis，Walking，Frappe，Diabetes，Criteo
搜索空间大小 N = 2000 个,
评价指标：balanced Acc & Spend time
Age-ASHA 探索阶段的数据采样率设置为 0.2，最大批次训练 300 batch，利用阶段的探索利用率设置为 1/5，即探索 2000 个，利用 400 个
对比基线：ASHA、Hyperband、BOHB
模型结构：MLP(512*6)


数据集 Adult  最大 4 epoch

Baseline            Balanced Acc    Time usage      Speed up
MLP                 0.821199        -               -
ASHA                0.829058        4961.36s        3.29 x
HyperBand           0.829098        8655.24         5.74 x
BOHB                0.829377        8978.04s        5.95 x
AgE-ASHA(ours)      0.828993        1508.49s        1.00 x
