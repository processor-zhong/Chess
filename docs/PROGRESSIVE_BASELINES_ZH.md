# 对齐feature：基础模型、类别权重与progress修复

当前主线选择性采用`feature@5f86a39`的数据设计与特征，不直接合并其删除原交接文件的仓库重构。
原始RF和固定20半回合的6／46列比较完整保留；本轮使用新10＋17列特征，独立结果目录，不能混用成绩。

## 对齐了哪些内容

同一份20,058行原始Lichess CSV，去重945行后19,113个唯一对局；保持feature分支的清洗规则。
按对局胜负分层，seed6001，70%训练、15%验证、15%测试：13,379／2,867／2,867盘。
先分对局，再回放棋谱，取最终SAN长度的10%、25%、50%、75%、100%位置。
极短棋局的取整位置可能重合，沿用原字典覆盖规则；生成66,346／14,200／14,179个局面。
同一盘棋的所有局面只能位于同一分区。

## 27列模型输入

所有模型直接输入以下相同27列，不包含概率融合层。

| 分组 | 输入 |
|---|---|
| 赛前10列 | white_rating、black_rating、rating_diff、rating_mean、rating_gap、rated_int、base_minutes、increment_seconds、log_base_minutes、white_elo_expectation |
| 子力7列 | pawn_diff、knight_diff、bishop_diff、rook_diff、queen_diff、material_balance、bishop_pair_diff |
| 位置10列 | mobility_diff、center_control_diff、king_shield_diff、doubled_pawn_diff、isolated_pawn_diff、passed_pawn_diff、castling_rights_diff、white_in_check、black_in_check、white_to_move |

差值均为白方减黑方。王屏障指王邻格己方兵数；机动性按原分支实现，分别设置行棋方后统计合法走法。
源实现见[src/progressive_features.py](../src/progressive_features.py)，契约见[feature_contract.json](../artifacts/progressive/feature_contract.json)。

## 修复progress输入泄漏

原feature分支在人工局面特征及概率融合输入中使用：

`progress = 当前半回合数 / 这盘棋最终的总半回合数`

最终总步数在真实预测时未知。新实现的`board_features(board)`不接受进度参数，也不生成进度列。
训练和预测通过明确的27列白名单取输入；ID、target、完整棋谱、采样百分比、当前ply和所有progress字段不进入学习器。
测试覆盖“增加或改变未来进度元数据不能改变模型输入”，并逐个核对保存模型的输入列。
当前交付仅含传统分类器；CNN、教师学生和进度辅助网络均不纳入提交。

修复范围是**模型输入中的未来信息**。采样百分比仍由最终棋谱长度确定，100%包含最后记录局面。
因此保留“回顾性百分比评估”的定位，不能称为固定前10步的实时预测，也不能将汇总准确率解释为开局准确率。

## 模型与none／balanced

每个学习模型都在两种设置下独立拟合、保存和报告；不是从两个设置中挑一个最好成绩。

| 模型 | 固定配方 | 不加权 | balanced实现 |
|---|---|---|---|
| Dummy | strategy=prior，学习训练类别先验 | 多数类预测及经验先验概率参照 | 不适用；等先验将不再是经验多数类参照 |
| LogisticRegression | C=0.1，max_iter=3000 | class_weight=None | class_weight=balanced |
| GaussianNB | var_smoothing=0.1，priors=None | 不传样本权重 | 向分类器传model__sample_weight |
| DecisionTree | max_depth=3，min_samples_leaf=5 | class_weight=None | class_weight=balanced |
| RandomForest | 200树，depth=8，leaf=10，sqrt，seed42 | class_weight=None | class_weight=balanced |
| HistGB | 150轮，lr=0.1，7叶，L2=0，关闭内部随机早停 | class_weight=None | class_weight=balanced |

按训练局面标签计算：`w_c = N_train / (3 × n_c_train)`。不使用验证或测试类别比例。
GaussianNB没有原生class_weight；使用同一组训练权重进行加权拟合，使经验类别先验均匀。
这不意味着所有算法的类别权重作用方式完全相同，且不等于声称输入满足高斯或条件独立假设。

LR、NB在训练Pipeline中拟合中位数插补、常量列移除与StandardScaler；树模型不缩放。
NB的插补和缩放保持训练集无权拟合，sample_weight仅用于分类器；与其他分类器的预处理设置一致。

这些参数来自旧20半回合比较的已选配方，属于参数迁移，本轮未重新调参。
不同算法的原搜索预算不等，不能称为它们各自的最优能力。

## 查看与复现

入口：[Progressive_Baselines.ipynb](../Progressive_Baselines.ipynb)，Run All只读保存结果，不训练。
完整结果：[RESULTS_ZH.md](../modelling_baselines/run_20261026_feature_01/RESULTS_ZH.md)。
其中包括验证/测试、每个时点、分类别指标、混淆矩阵、逐局面概率及完整参数。

Python3.12和固定requirements.txt依赖，不需要PyTorch。

```bash
python -m pip install -r requirements-notebook.txt
python verify_progressive_baselines.py --reload-models
python -m unittest discover -s tests -p "test_progressive*.py" -v
python run_progressive_baselines.py --output modelling_baselines/local_progressive_reproduce
```

已有结果目录禁止覆盖。缓存缺失时从原始CSV重建，不需要网络下载；数据源与代码哈希均保留。
Windows深层路径由新脚本内部使用扩展路径处理；Git克隆时可使用`git -c core.longpaths=true clone ...`。

## 评估边界

准确率、Macro-F1、Balanced Accuracy、Log Loss、宏平均OvR AUC，以及和棋Precision/Recall/F1都保留。
不能仅凭提高和棋Recall判断更好，也不能仅凭总体Accuracy判断概率质量。
同一盘棋通常贡献5个相关局面，详细结果同时列出各时点，不能按独立局面数夸大独立样本量。
历史测试已在前序探索中查看，旧参数也源于同源数据；这些是开发性比较，不是全新盲测或显著性结论。
