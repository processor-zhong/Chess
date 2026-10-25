# 基础模型比较和调参

本扩展负责比较不同算法及有限调参，沿用20个半回合观察时点、原始时间分区、黑胜／和棋／白胜三分类。
使用训练11,936局、验证2,549局；不读取测试文件、不读取旧测试预测，也不重新评估测试集。

## 模型及理由

| 模型 | 用途 | 预处理／限制 |
|---|---|---|
| DummyClassifier | 训练多数类参照，判断其他模型是否有额外预测能力 | 不调参；比只引用33%均匀随机更合适 |
| LogisticRegression | 线性判别基线，检查评分差与局面特征的线性信息 | 折内标准化；搜索C和类别权重 |
| GaussianNB | 简单概率模型，与线性及树模型形成不同假设的对照 | 折内标准化；高斯及条件独立假设可能不成立 |
| DecisionTreeClassifier | 单棵树对照及过拟合诊断 | 搜索深度、叶样本数及类别权重；不标准化 |
| RandomForestClassifier | 非线性树集成，复用已有RF模型族 | 搜索深度、叶样本数及类别权重 |
| HistGradientBoostingClassifier | 顺序拟合误差的树集成，与RF进行算法比较 | 不标准化；关闭随机内部早停，保持时间协议 |

全部Pipeline在每折训练部分拟合中位数插补和VarianceThreshold。导出的输入仍为6列或46列。
StandardScaler不意味着原始数据变成高斯分布。NB先验设为均匀不等同于class_weight。

## 对照设计

两套输入对每个模型都保留相同的对局：

- pregame：原赛前6列。
- combined：赛前6列＋前20个半回合提取的40列。

“初始”表示未经本轮搜索的仓库starter设置，不是宣称全部使用sklearn默认参数：
LR的C=1、max_iter=3000；RF为300棵树、min_samples_leaf=5；HistGB为150轮、15个叶节点、关闭内部早停。
NB使用默认var_smoothing和经验类别先验。

## 搜索与选择

| 模型 | 搜索内容 | 候选数／每套特征 |
|---|---|---:|
| LR | C=0.01/0.1/1/10；class_weight=None/balanced | 8 |
| GaussianNB | var_smoothing=1e-9/1e-6/1e-3/0.1；经验／均匀先验 | 8 |
| 决策树 | 深度3/5/8/12/不限；叶样本5/20；权重None/balanced | 20 |
| RF | 200树；深度8/16/不限；叶样本3/10；权重None/balanced | 12 |
| HistGB | 150轮、学习率0.1；叶节点7/15；L2=0/10；权重None/balanced | 8 |

每个候选在训练集内做三折扩展时间CV，整UTC日划分，边界间隔24小时。
按平均Macro-F1选参；并列按较低CV Log Loss，再按候选序号选择。
每个模型、每套特征独立选择一个候选，冻结文件保存后才加载验证标签。
初始配置与搜索配置完全相同时复用该配置的CV拟合，表中reused_cv会明确标识。

搜索规模不同且范围有限，不代表比较了各算法的全局最优表现。
原RF历史实验为两套信息集选择共享参数，本轮分别选参，不能把二者的差异全部归因为算法。

## 输出与解释

入口：[Baseline_Model_Comparison.ipynb](../Baseline_Model_Comparison.ipynb)。Run All只读取保存结果及图，不启动训练。

正式训练入口：`run_baseline_models20.py`；实现：`src/baseline_models20.py`。
图表和核对实现：`src/baseline_report20.py`；独立核对入口：`verify_baseline_models20.py`。

结果目录：`modelling_baselines/run_20261025_20ply_01/`，包括18个模型配置的验证结果、逐局概率、混淆矩阵、训练集拟合模型与参数记录。
重点同时查看Macro-F1、Accuracy、各类别Precision/Recall/F1、Balanced Accuracy、Log Loss、OvR AUC与训练／验证差距。
调参可能提高Macro-F1但降低准确率；不能只筛选改善的指标进行汇报。

所有模型文件均只在训练集拟合；不要把其验证指标与train+validation重训模型的分数混用。
验证集经过多轮探索，且历史测试期已经被先前实验查看，本轮结果仅为开发比较。
不删除和棋、不输入棋手ID，不依据已有测试表现选择模型。

## 本轮观察

46列特征的固定验证结果如下，均不是测试成绩：

| 模型 | 初始Accuracy → 调参Accuracy | 初始Macro-F1 → 调参Macro-F1 | 调参和棋Recall |
|---|---|---|---:|
| 逻辑回归 | 62.18% → 62.10% | 0.4241 → 0.4236 | 0% |
| GaussianNB | 58.61% → 58.81% | 0.4147 → 0.4161 | 2.50% |
| 决策树 | 52.65% → 60.22% | 0.3858 → 0.4088 | 0% |
| 随机森林 | 61.40% → 56.18% | 0.4189 → 0.4443 | 23.33% |
| HistGB | 61.63% → 61.63% | 0.4207 → 0.4207 | 0% |

多数类基线Accuracy=48.61%、Macro-F1=0.2181。
此Dummy仅作为类别预测基线，most_frequent给出0/1概率，故其很大的Log Loss不宜用来代表合理的概率基线。

本轮没有出现所有指标同时显著改善的模型。逻辑回归准确率较高，但没有识别出和棋。
RF调参后识别出120场和棋中的28场，同时把276场非和棋误判为和棋，和棋Precision仅9.21%；因此不能仅报告Recall改善。
RF的46列初始训练Accuracy=86.50%、验证61.40%，调参后训练65.15%、验证56.18%，正则化缩小了拟合差距，但未提高验证Accuracy。
GaussianNB加入开局信息后Log Loss仍大于1，概率质量弱于LR和HistGB，不能只凭少量准确率变化认为假设成立。

同一队列的6列与46列比较支持“增加开局信息对部分模型有帮助”的描述；便利抽样、重复棋手及单一历史验证期限制了推广。
现有结果不支持宣称复杂模型必然优于线性模型，也没有证明当前特征的预测上限。

## 复现

单棵决策树入口为`run_decision_tree20.py`，结果见`modelling_baselines/run_20261025_tree_01/`。
默认树在46列输入上的实际深度为38，训练Accuracy=99.99%、验证52.65%；
训练CV选定深度3、叶样本至少5、不加权后，训练63.56%、验证60.22%，显示深度约束缓解过拟合。
决策树同样只按训练CV选参；各算法搜索规模有限且不同。
统一汇总运行`python compare_saved_models20.py`，默认仅包括传统模型，不重新训练。
神经网络探索保留在原工作目录的独立运行记录中，本次提交不包含这些实验。

Python 3.12及固定依赖见requirements.txt。Windows建议把环境放在短路径，避免深层课程文件夹触发包安装路径长度限制。

```bash
python -m pip install -r requirements-notebook.txt
python -m unittest discover -s tests -v
python run_baseline_models20.py --output modelling_baselines/local_my_experiment
python verify_baseline_models20.py --output modelling_baselines/local_my_experiment --reload-models
```

已有目录会被拒绝，防止覆盖结果；local_*不默认提交Git。
`--reload-models`仅复现验证集概率，不打开测试集。新增流程没有finalize选项。

冻结结果目录、已执行Notebook和仓库快照清单由.gitattributes禁用Git换行转换，以保持克隆后的哈希一致。

查看仓库保存结果无需加载joblib；加载模型时应使用与训练相容的固定依赖。
