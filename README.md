# Chess outcome prediction after 10 full moves

CA6001 group handover: **Outcome prediction after 10 full moves (20 plies)**.

使用双方各前10步结束时的信息，预测 **Black win / Draw / White win**。保留“赛前6列”与“赛前＋开局46列”的特征实验。**此仓库不包含Stockfish扩展或旧10半回合实验。**

## 本次提交：基础模型比较与调参

多数类基线、逻辑回归、高斯朴素贝叶斯、单棵决策树、随机森林、直方图梯度提升。
对比赛前6列和赛前＋开局46列；训练内三折时间CV选参，固定验证集评价。
本次主线不包含神经网络。初始配置是仓库starter设置，不全部等同于sklearn默认参数。

1. [提交说明与报告摘要](docs/SUBMISSION_ZH.md)：范围、结果、RF来源和文件清单。
2. [Baseline_Model_Comparison.ipynb](Baseline_Model_Comparison.ipynb)：已执行的比较、调参、三分类混淆矩阵和ROC。
3. [统一比较表与图](modelling_baselines/comparison_20ply/COMPARISON_ZH.md)。
4. [方法与参数网格](docs/BASELINE_MODELS_ZH.md)。

```bash
python verify_snapshot.py
python verify_baseline_models20.py --reload-models
python verify_decision_tree20.py --reload-models
python -m unittest discover -s tests -v
```

## 原仓库已经包含的随机森林

原提交`0bf217e`已经提供RF训练、调参、模型和评估；相关代码及`modelling_rf/`结果保留。
新增RF比较使用另一套选参协议，具体差异见[提交说明](docs/SUBMISSION_ZH.md)。
原入口为[Modelling_Evaluation.ipynb](Modelling_Evaluation.ipynb)和[原RF说明](docs/MODELLING_EVALUATION.md)。
下方历史测试成绩不属于新增多模型比较的验证成绩。

## 环境

推荐 **Python 3.12**（本次整理验证3.12.15；固定依赖要求Python >=3.11）。

```bash
git clone https://github.com/processor-zhong/Chess.git
cd Chess
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-notebook.txt
jupyter lab
```

只跑脚本可安装 `requirements.txt`。模型文件和数据都已包含，不需要重新下载数据或重新训练才能查看结果。

## 已有随机森林历史测试结果

同一批2,565场测试对局；两个信息集使用相同随机森林参数。

| 指标 | 赛前6列 | 赛前＋开局46列 |
|---|---:|---:|
| Accuracy | 55.95% | 61.64% |
| Macro-F1 | 0.4177 | 0.4495 |
| Balanced Accuracy | 41.92% | 45.02% |
| Draw recall | 8.33% | 5.30% |
| Log Loss，越低越好 | 0.9101 | 0.8836 |

![Feature information experiment](modelling_rf/run_20261025_20ply_01/figures/test_feature_comparison.png)

整体预测改善，但132场和棋只识别7场，不能只看Accuracy。该时点实验是在更早一轮结果被查看后开展的探索；沿用的历史测试时间段不属于全新盲测。未进行显著性检验。

## 目录

```text
Modelling_Evaluation.ipynb        已执行结果查看入口
Baseline_Model_Comparison.ipynb  新增多模型开发比较入口
run_baseline_models20.py         初始模型、训练内调参、固定验证评估
verify_baseline_models20.py      不读取测试集的结果核对
README.md
requirements*.txt
run_plan_a20.py                  生成20半回合输入
validate_plan_a20.py              数据/棋谱/接口检查
run_rf_modelling20.py             训练内CV、冻结、最终评估
rebuild_data.py                   从原始CSV完整重建（临时目录隔离旧格式）
predict_opening.py                用一场新的20半回合棋谱预测
verify_snapshot.py               核对共享文件哈希和保存的测试指标
src/                             清洗、棋局特征、模型与指标实现
data/raw/                        原始CSV、来源和许可说明
data/processed/                  去重后的完整审计表，不可整表喂给模型
data/splits/split_manifest.csv   原固定时间分区与边界排除
data/plan_a20/splits/             本版训练/验证 X、y、meta
data/plan_a20/holdout/            本版冻结测试 X、y、meta
tables/plan_a20/                 特征字典、排除日志、训练集统计
artifacts/plan_a20/              特征契约、来源哈希、88项检查结果
modelling_rf/run_20261025_20ply_01/
  models/                        两种信息集的训练模型和最终重训模型
  tables/                        CV、验证、测试、分类别指标及逐场预测
  figures/                       指标图、混淆矩阵、重要性图
  artifacts/                     流程、冻结配置和结果验证记录

modelling_baselines/run_20261025_20ply_01/
  tables/ figures/ models/ artifacts/  新多模型开发结果（无测试成绩）
docs/                            数据协议、操作和结果说明
```

## 复现与继续工作

```bash
python verify_snapshot.py
python validate_plan_a20.py
# 新实验使用独立目录；local_*默认不进入Git
python run_rf_modelling20.py --stage tune --output modelling_rf/local_my_experiment
# 只有方案冻结并且小组决定正式评估后，才执行：
python run_rf_modelling20.py --stage finalize --output modelling_rf/local_my_experiment
```

不要覆盖已有结果，不要根据测试成绩回头调参，不要将meta中的ID/FEN或完整审计表中的赛后字段输入模型。不要为了提高成绩删除和棋。更改特征需同步更新预处理和契约。

## 对新的开局预测

提供恰好20个半回合的合法SAN棋谱，例如：

```bash
python predict_opening.py --white-rating 1600 --black-rating 1500 --base-minutes 10 --increment-seconds 0 --moves "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7"
```

输出预测类别及三类概率，不会读取测试集。

## 数据来源

[Chess Game Dataset (Lichess), Mitchell J / datasnaek](https://www.kaggle.com/datasets/datasnaek/chess)，发布页标注CC0。
原始20,058行，清洗后19,111个对局ID。本版过滤后训练11,936、验证2,549、测试2,565场。详见 [数据协议](docs/DATA_PROTOCOL.md) 和 [原始来源](data/raw/SOURCE.md)。

仓库未替团队代码指定额外的开源许可；数据仍按其发布者标注的许可使用。此交接不包含课程正式报告。
