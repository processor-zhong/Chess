# 建模与评估记录

只训练RandomForestClassifier，对比相同样本上的赛前6列与赛前＋开局46列。

## 训练和选择

12组共享参数候选；每组在两个信息集上分别进行3折扩展时间CV，所有拟合步骤只用每折训练部分。
按两组平均CV Macro-F1取前3组，再按两组平均验证Macro-F1选共享参数。
冻结200树、最大深度16、叶节点最少10样本、max_features=sqrt、class_weight=balanced、random_state=42。
预测为概率argmax，未根据测试标签调阈值。
验证阶段选择的主版本为combined，验证Macro-F1=0.4467，对照pregame=0.4251。
按事先固定政策在训练+验证14,485场上重新拟合两版完整Pipeline，再在2,565场测试对局上统一评分。

## 最终测试

| 指标 | Pregame | Combined |
|---|---:|---:|
| Accuracy | 0.559454 | 0.616374 |
| Macro-F1 | 0.417667 | 0.449503 |
| Balanced Accuracy | 0.419208 | 0.450219 |
| Draw Recall | 0.083333 | 0.053030 |
| Log Loss | 0.910133 | 0.883630 |

开局信息在这一设置中改善整体预测和概率损失，但和棋Recall下降：132场只识别7场。没有作显著性检验，不能声称稳定或因果改进。
验证置换重要性使用仅训练集拟合的模型，测试集不参与特征选择。

## 已保存证据

modelling_rf/run_20261025_20ply_01/artifacts/中有protocol、frozen_config、development_complete、test_access、final_evaluation_complete与result_verification。
原数据验证88项通过，最终结果核对25项通过。预测CSV可用于复核分数，无需再次给测试模型评分。
目录中checkpoint_common_cohort_metrics.csv是历史时点探索的记录，只固定了共同测试样本，未控制训练队列或参数；不把它当作预测时点净效果实验。

## 复现说明

查看README中的tune/finalize命令，新运行使用独立目录，默认local_*不提交Git。
最终joblib要求相容的固定依赖环境；文件是本项目生成的可加载Pipeline，不需手动重建预处理器。
