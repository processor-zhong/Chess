# 传统模型模块的整合接口

本分支从feature@5f86a39建立，没有将另一条分支的历史数据目录和旧20半回合流程强行合并。
新增baseline_models.py及独立结果，原Notebook只修复progress输入并增加一个模型调用区块。

## 两个数据对象

- games：原Notebook清洗、重置索引并划分后的唯一对局表，含赛前信息。
- snapshots：原Notebook回放后生成的局面表，含game_index、game_id、split、target、checkpoint以及17列局面特征。

assemble_inputs(games, snapshots)按game_index连接赛前10列，返回同序的27列预测变量。
ID、目标、棋谱、checkpoint与ply均不进入模型。若任一数据对象含progress或actual_progress，接口直接报错。

## 模型API

```python
from baseline_models import compare_baselines, assemble_inputs, predict_probabilities
registry, metrics = compare_baselines(games, snapshots)
X = assemble_inputs(games, snapshots)
p = predict_probabilities(registry["logistic_none"], X)
```

输出概率列顺序是black、draw、white。注册表键为模型族加权重，例如random_forest_balanced。
模型族：dummy、logistic、gaussian_nb、decision_tree、random_forest、gradient_boosting。
这些是直接27列分类器，不能未经调整就替换原来只接收17列的position-only模型。
本模块不修改原Notebook的模型选择变量，团队可独立决定后续使用哪条预测路径。

需要保存时：compare_baselines(games, snapshots, output_dir="results/local_my_run")。
已有目录禁止覆盖；results/local_*被忽略。保存模型可直接joblib.load后用predict_probabilities调用。

## 参数和类别权重

| 模型 | 固定参数 |
|---|---|
| LR | C=0.1，max_iter=3000 |
| GaussianNB | var_smoothing=0.1；balanced时向分类器传训练sample_weight |
| 决策树 | depth=3，leaf=5 |
| RF | 200树，depth=8，leaf=10，sqrt |
| HistGB | 150轮，lr=0.1，7叶，L2=0，关闭随机内部早停 |

模型seed42；原数据分层seed6001。none/balanced均独立报告；权重为N_train/(3*n_class_train)。
插补、常量列处理及LR/NB标准化只在训练集拟合；NB的sample_weight只作用于分类器。
参数来自此前基础模型分支的固定配方，不是这套新特征上重新调参的最优解。

## progress删除范围

1. board_features只接收board，删除progress特征；快照不再保存actual_progress。
2. MATERIAL_FEATURES删除progress，POSITIONAL_FEATURES剩17列。
3. SnapshotDataset只返回board和target；PositionCNN.forward只接收board，首个全连接层从65改为64。
4. predict_cnn及CNN训练循环删除进度参数；stack_inputs只拼六个对数概率。
5. 不实现教师、辅助进度损失或预测进度头。Notebook旧输出全部清空。

原CNN未重新训练，本次只验证其删列后的前向接口。英文Word报告和原figures均属旧配置，需后续完整运行后更新。
全部Notebook代码均编译检查，传统模块在修改后的Notebook数据上重新训练，132行指标与基础模型分支一致。

百分比仍仅用于选取回顾性评估位置；最终长度未知的问题使它不等同于实时固定步数评估。
测试时期已被先前探索查看，结果不应称为新的独立盲测或显著性结论。

?????requirements-baselines.txt?????????uv.lock??????????????????????
