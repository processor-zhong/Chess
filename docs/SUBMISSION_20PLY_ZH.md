# 本次提交：传统模型对比与调参

本交付完成小组的基础模型比较与有限调参部分，包含可复现代码、已执行Notebook、模型、逐局概率和图表。
神经网络探索不纳入本次提交包；既有探索文件保留在本地供以后参考。
本文是该部分的报告素材与交接说明，课程完整报告仍应按小组模板汇总各成员工作。

## 任务和数据

在双方各走10步，即20个半回合后，预测黑胜、和棋、白胜。
比较相同对局上的两套输入：赛前6列，以及加上40列开局汇总后的46列。
赛前列为rating_diff、avg_rating、abs_rating_diff、base_time_min、increment_seconds、rated。
开局特征由前20个半回合提取，包含子力、发展、中心控制、王位置、王车易位、吃子和将军等；
完整字段见`tables/plan_a20/feature_dictionary.csv`。
棋手ID和对局ID不输入模型，20步之后的实际走法及赛后信息不作为特征。

沿用原数据去重、时间分区、24小时间隔和未结束对局过滤规则。
训练11,936局，验证2,549局，目标保留和棋；所有算法使用同一队列。
新增模型仅在训练集拟合，没有产生新的测试成绩。

## 对比与调参方法

多数类Dummy不调参；其余比较逻辑回归、GaussianNB、决策树、RF、HistGB。
每套特征、每个模型独立进行训练内三折扩展时间CV，整UTC日划分并留24小时间隔。
预处理在每折训练部分拟合：中位数插补、常量列移除；LR和NB额外标准化。
按平均CV Macro-F1选择参数，并列按较低CV Log Loss和候选序号；冻结参数后评估验证集。
完整网格见[方法说明](BASELINE_MODELS_ZH.md)。决策树20组、RF12组、其他学习器各8组，搜索预算不同且有限。
“初始”采用仓库starter设置，决策树仅固定随机种子，其余不应笼统称为sklearn默认参数。

## 46列输入的验证结果

以下为训练CV选定版本，非按验证成绩挑选初始／调参版本。完整6列及46列、初始及调参表均保留。

| 模型 | Accuracy | Macro-F1 | 和棋Precision | 和棋Recall |
|---|---:|---:|---:|---:|
| 多数类基线 | 48.61% | 0.2181 | 0% | 0% |
| 逻辑回归 | 62.10% | 0.4236 | 0% | 0% |
| GaussianNB | 58.81% | 0.4161 | 11.11% | 2.50% |
| 决策树 | 60.22% | 0.4088 | 0% | 0% |
| 随机森林 | 56.18% | 0.4443 | 9.21% | 23.33% |
| HistGB | 61.63% | 0.4207 | 0% | 0% |

LR具有本轮最高的验证Accuracy，但没有预测和棋。RF取得最高Macro-F1，同时牺牲Accuracy；
120场和棋识别28场，276场非和棋被误报，不能只引用Recall上升。
LR和HistGB的Log Loss约0.78，不能由Accuracy约60%推断AUC必然接近0.5。
例如LR的三分类宏平均OvR AUC约0.6801，RF约0.6632；AUC评价概率排序，与分类准确率含义不同。

决策树是明确的过拟合示例：初始深度38、训练Accuracy99.99%、验证52.65%；
训练CV选择深度3、叶样本至少5、不加权后，训练63.56%、验证60.22%。
RF从初始训练86.50%／验证61.40%，变为调参后65.15%／56.18%；拟合差距减小，但验证准确率未提高。
调参不保证所有指标改善，结果已完整保留。

## 随机森林原来就有吗？

**有。** GitHub原仓库提交`0bf217e`（Add reproducible 20-ply random forest team handover）已包含：

- `src/rf_experiment20.py`、`run_rf_modelling20.py`：RF训练、调参和评估。
- `modelling_rf/run_20261025_20ply_01/`：已保存的参数、模型、CV／验证／历史测试结果。
- `Modelling_Evaluation.ipynb`：原RF结果查看入口。

| 区别 | 原仓库RF | 新增统一比较中的RF |
|---|---|---|
| 参数选择 | 两套特征共享参数；训练CV筛选后结合验证选定 | 每套特征独立按训练CV选定 |
| 46列所选参数 | 200树，深度16，叶样本10，balanced | 200树，深度8，叶样本10，balanced |
| 本文对比使用的成绩 | 原记录单独保留 | 仅训练拟合后的固定验证成绩 |
| 最终重训 | 原流程训练＋验证重训并已有历史测试成绩 | 本轮不重训训练＋验证、不新增测试评分 |

因此新增贡献是扩展算法、提供统一比较与调参诊断，而不是首次实现随机森林。
原RF的历史测试Accuracy61.64%与新RF验证Accuracy56.18%来自不同评估阶段和选参协议，不能直接当成退步幅度。

## 文件与运行

先打开根目录`Baseline_Model_Comparison.ipynb`；已经保留执行输出，Run All只读取结果，不训练、不读取测试集。
`modelling_baselines/comparison_20ply/COMPARISON_ZH.md`提供统一表图，CSV保留22个初始／调参评估条目。
`run_20261025_20ply_01/`和`run_20261025_tree_01/`保留CV、逐场预测、混淆矩阵、模型与来源哈希。
原RF在`modelling_rf/`独立保留。
`build_submission20.py`按明确清单导出传统模型材料，排除神经网络、环境、缓存和Git元数据。
在仓库中可运行`python build_submission20.py --output ../CA6001_Basic_Models_Submission.zip`，已有压缩包不会被覆盖。

Python3.12，安装固定依赖后核对：

```bash
python -m pip install -r requirements-notebook.txt
python verify_snapshot.py
python verify_baseline_models20.py --reload-models
python verify_decision_tree20.py --reload-models
python -m unittest discover -s tests -v
```

`verify_snapshot.py`校验文件清单并重算原RF已保存预测的历史指标，不训练或重新预测测试集。
新增模型的两个核对脚本只读取训练／验证相关材料；可加载训练模型重现验证概率。
重新做实验请使用新目录，避免覆盖冻结结果：

```bash
python run_baseline_models20.py --output modelling_baselines/local_reproduce
python run_decision_tree20.py --output modelling_baselines/local_tree_reproduce
python compare_saved_models20.py
```

## 结论边界

复杂模型并未在所有指标上超过线性基线；类别不平衡使和棋识别仍然困难，但没有证明和棋不可预测或现有特征达到上限。
验证集经历多轮探索，历史测试期也已经被先前实验查看，因此本轮是开发比较，没有新的独立盲测或显著性结论。
不删除罕见开局或和棋换取高分；不把参考项目存在重复对局泄露的成绩当成同条件基准。
