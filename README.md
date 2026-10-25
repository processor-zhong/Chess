# Progressive Chess Outcome Prediction — integration branch

本分支基于`feature@5f86a39`，保留原Notebook的分析结构，补入可独立调用的传统模型模块，方便后续整合。

## progress已从全部模型输入中删除

- 人工局面特征：删除progress参数、字段和特征列，剩17列。
- 原CNN：删除进度数据项和参数，只输入17个棋盘平面；池化后的64维向量直接进入分类器。
- 原概率融合：删除actual_progress，只输入两组模型的六个对数概率。
- 不包含教师、蒸馏或进度辅助预测方案；真实progress不是任何模型的训练或预测输入。

原Notebook所有已保存输出已清空，避免沿用修改前含泄漏的成绩。
原英文报告和figures目录保留为**旧版本参考材料**，其中数值不代表本修订版结果。

采样百分比只作为回顾性评估分组。它仍使用最终长度选择位置，100%包含最后局面，不能将汇总成绩解释为固定开局能力。

## 新增模型模块

`baseline_models.py`直接使用Notebook准备好的`games`与`snapshots`，把10列赛前＋17列局面合成27列。
LR、GaussianNB、决策树、RF、HistGB分别拟合none/balanced，另加经验先验Dummy，共11个模型。
它独立返回模型字典和结果，不改写原Notebook的best_handcrafted、CNN选择或stacker变量。

Notebook已经在棋谱回放之后加入以下调用：

```python
from baseline_models import compare_baselines, assemble_inputs, predict_probabilities

models, results = compare_baselines(games, snapshots)
X = assemble_inputs(games, snapshots)
probabilities = predict_probabilities(models["gradient_boosting_none"], X)
```

## 不加权模型：回顾性测试汇总

| 模型 | Accuracy | Macro-F1 | 和棋Precision | 和棋Recall | 和棋F1 | Log Loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 多数类基线 | 49.86% | 0.2218 | 0.00% | 0.00% | 0.0000 | 0.8486 |
| 逻辑回归 | 69.23% | 0.4716 | 0.00% | 0.00% | 0.0000 | 0.6814 |
| 高斯朴素贝叶斯 | 66.75% | 0.4913 | 9.64% | 7.99% | 0.0874 | 1.3270 |
| 决策树 | 67.90% | 0.4620 | 0.00% | 0.00% | 0.0000 | 0.7308 |
| 随机森林 | 69.06% | 0.4703 | 0.00% | 0.00% | 0.0000 | 0.6841 |
| HistGB | 69.23% | 0.4737 | 13.33% | 0.30% | 0.0059 | 0.6764 |

## balanced模型：回顾性测试汇总

| 模型 | Accuracy | Macro-F1 | 和棋Precision | 和棋Recall | 和棋F1 | Log Loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 逻辑回归 | 56.21% | 0.4661 | 5.89% | 34.39% | 0.1005 | 0.9170 |
| 高斯朴素贝叶斯 | 51.28% | 0.4426 | 5.78% | 42.68% | 0.1019 | 1.6227 |
| 决策树 | 46.39% | 0.4058 | 5.34% | 47.66% | 0.0960 | 0.9733 |
| 随机森林 | 56.25% | 0.4675 | 5.82% | 34.54% | 0.0996 | 0.8959 |
| HistGB | 60.13% | 0.4886 | 8.52% | 35.44% | 0.1374 | 0.8675 |

balanced只按训练标签计算；NB通过sample_weight，其余学习器通过class_weight。Dummy balanced不适用。
结果来自相同2,867盘测试棋的14,179个局面，包括最后位置；模型参数为固定配方迁移，未在新测试上调参。
历史测试已经被查看，本轮是探索性比较。

## 结果与整合说明

- [整合接口和移除progress的具体改动](docs/BASELINE_INTEGRATION.md)
- [完整验证／测试及各时点结果](results/baselines_20261026/README.md)
- [全部指标CSV](results/baselines_20261026/tables/metrics.csv)
- [分类别指标](results/baselines_20261026/tables/per_class.csv)
- [保存模型](results/baselines_20261026/models)与[参数](results/baselines_20261026/artifacts/parameters.json)
- [原Notebook](CA6001_chess_outcome_prediction.ipynb)：模型输入已修复，原CNN未在本次重新训练。

```bash
# 仅使用/核对传统模型，不需要PyTorch：
python -m pip install -r requirements-baselines.txt
python verify_baselines.py --reload-models
python -m unittest discover -s tests -v
```

原Notebook全流程仍沿用Python3.11与uv.lock：`uv sync --locked`后运行`uv run jupyter lab`。
数据仍在data/games.csv，不在版本控制中；缺失时Notebook按原方式下载Kaggle数据。
新增模型的调用示例不写文件；如需保存新实验，给compare_baselines传入一个不存在的output_dir。

原报告和图表需要在团队后续完整重跑后统一更新，不能继续引用旧的81.2%作为本修订版结果。
