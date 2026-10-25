# 当前提交：对齐feature并修复progress输入泄漏

本次主线为新27列输入下的传统模型比较。算法及权重设置、结果表和详细链接见根目录README。

- 数据、特征与回顾性采样对齐feature@5f86a39；模型输入删除真实progress，27列白名单取值。
- 5种学习器各none/balanced两个版本，加Dummy经验先验参照，共11个模型；不包括神经网络。
- GaussianNB使用训练sample_weight；其余学习器用class_weight；Dummy balanced不适用。
- 所有参数为旧配方迁移，没有在新特征上重新调参；旧调参记录保留。
- 全部验证/测试及分时点结果、逐类别指标、预测概率、混淆矩阵、模型和校验记录均保存。
- 百分比采样仍依赖最终长度；包括终局的汇总分数不能当成固定开局能力。历史测试已查看，本轮为探索性比较。

阅读顺序：README → [当前方法说明](PROGRESSIVE_BASELINES_ZH.md) → 根目录Progressive_Baselines.ipynb。
完整结果见[本轮报告](../modelling_baselines/run_20261026_feature_01/RESULTS_ZH.md)。
原固定20半回合交付见[历史说明](SUBMISSION_20PLY_ZH.md)，数据和结果完整保留。
本文是模型部分交接材料，课程完整报告仍需小组合并。
