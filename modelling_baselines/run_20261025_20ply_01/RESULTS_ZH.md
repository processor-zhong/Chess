# 基础模型对比与有限调参：20个半回合

使用同一训练集11,936局与验证集2,549局，保留黑胜、和棋、白胜三分类。
赛前6列与赛前＋开局46列分别比较；本轮没有读取测试集或旧测试预测。

## 验证结果

| 输入 | 模型 | 阶段 | Accuracy | Macro-F1 | 和棋Precision | 和棋Recall | Log Loss |
|---|---|---|---:|---:|---:|---:|---:|
| pregame | 多数类基线 | 初始 | 48.61% | 0.2181 | 0.00% | 0.00% | 18.5238 |
| pregame | 逻辑回归 | 初始 | 59.71% | 0.4067 | 0.00% | 0.00% | 0.7898 |
| pregame | 逻辑回归 | CV调参后 | 59.71% | 0.4067 | 0.00% | 0.00% | 0.7898 |
| pregame | 高斯朴素贝叶斯 | 初始 | 59.20% | 0.4188 | 9.09% | 2.50% | 0.8403 |
| pregame | 高斯朴素贝叶斯 | CV调参后 | 59.20% | 0.4188 | 9.09% | 2.50% | 0.8402 |
| pregame | 随机森林 | 初始 | 58.06% | 0.3954 | 0.00% | 0.00% | 0.8131 |
| pregame | 随机森林 | CV调参后 | 55.94% | 0.4248 | 6.88% | 12.50% | 0.9058 |
| pregame | 直方图梯度提升 | 初始 | 59.44% | 0.4103 | 50.00% | 0.83% | 0.8100 |
| pregame | 直方图梯度提升 | CV调参后 | 59.83% | 0.4077 | 0.00% | 0.00% | 0.7877 |
| combined | 多数类基线 | 初始 | 48.61% | 0.2181 | 0.00% | 0.00% | 18.5238 |
| combined | 逻辑回归 | 初始 | 62.18% | 0.4241 | 0.00% | 0.00% | 0.7795 |
| combined | 逻辑回归 | CV调参后 | 62.10% | 0.4236 | 0.00% | 0.00% | 0.7795 |
| combined | 高斯朴素贝叶斯 | 初始 | 58.61% | 0.4147 | 8.57% | 2.50% | 1.4294 |
| combined | 高斯朴素贝叶斯 | CV调参后 | 58.81% | 0.4161 | 11.11% | 2.50% | 1.2998 |
| combined | 随机森林 | 初始 | 61.40% | 0.4189 | 0.00% | 0.00% | 0.7826 |
| combined | 随机森林 | CV调参后 | 56.18% | 0.4443 | 9.21% | 23.33% | 0.9837 |
| combined | 直方图梯度提升 | 初始 | 61.63% | 0.4207 | 0.00% | 0.00% | 0.7920 |
| combined | 直方图梯度提升 | CV调参后 | 61.63% | 0.4207 | 0.00% | 0.00% | 0.7828 |

## 选定参数

每个模型、每套输入独立选择，全部在打开验证标签之前冻结。

- pregame / 逻辑回归：`{"C": 1.0, "class_weight": null}`
- pregame / 高斯朴素贝叶斯：`{"var_smoothing": 0.001, "priors": null}`
- pregame / 随机森林：`{"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced"}`
- pregame / 直方图梯度提升：`{"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 7, "l2_regularization": 10.0, "class_weight": null}`
- combined / 逻辑回归：`{"C": 0.1, "class_weight": null}`
- combined / 高斯朴素贝叶斯：`{"var_smoothing": 0.1, "priors": null}`
- combined / 随机森林：`{"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced"}`
- combined / 直方图梯度提升：`{"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 7, "l2_regularization": 0.0, "class_weight": null}`

## 如何解释

- 主要选参指标是训练内三折时间CV的平均Macro-F1；并列时用CV Log Loss和候选序号打破平局。验证结果不用于重新选参。
- CV选参不保证验证准确率、和棋Recall或Macro-F1全部上升，调参失败也保留并报告。
- 初始设置沿用仓库starter，不能全部称为sklearn默认值。随机森林初始300树、叶样本5；梯度提升初始150轮、15叶、关闭随机内部早停。
- 逻辑回归与高斯朴素贝叶斯在每折内标准化；树模型不标准化。全部模型在每折内拟合插补和常量特征移除。
- GaussianNB的均匀先验与class_weight不同。标准化不会使数据自动服从高斯分布；局面计数和派生变量可能不满足条件独立假设。
- 搜索量：LR 8组、GaussianNB 8组、RF 12组、HistGB 8组，每组每个信息集3折。搜索预算有差别，不能声称找到了各算法全局最优参数。
- 最终模型文件只在训练集拟合，用于验证解释与复现；没有在train+validation重训，也没有新的测试成绩。
- 原仓库RF为两套特征选择共享参数；本轮为每个模型和信息集分别按训练CV选择。它们属于不同选参协议，不能把分数差全部归因为算法。
- 验证集经过此前多轮探索，历史测试也已被查看；本轮是开发比较，不能宣称全新盲测结论或显著提升。
- 均匀随机准确率33%不足以作为唯一参照；表中提供实际训练多数类模型。

## 文件

- tables/validation_metrics.csv：总体指标、训练分数、CV分数、参数和耗时。
- tables/tuning_changes.csv：同模型初始到调参后的变化；Log Loss变化越负越好。
- tables/cv_summary.csv、cv_fold_metrics.csv：全部候选的训练CV记录。
- tables/*_predictions.csv、*_confusion.csv：逐局三类概率与混淆矩阵。
- figures/：模型比较、混淆矩阵、拟合差距、CV搜索和ROC图。
- artifacts/：流程、源文件哈希、冻结参数和核对记录。
- models/：每个初始／调参模型的训练集Pipeline。

本文件由实验脚本生成；复现请使用新的输出目录，不能覆盖已有结果。
