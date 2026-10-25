# 小组后续操作说明

## 推荐阅读顺序

README → Modelling_Evaluation.ipynb → MODELLING_EVALUATION.md → DATA_PROTOCOL.md。
Notebook是已执行的结果查看器；实际训练和评估代码在src/rf_experiment20.py，调用入口为run_rf_modelling20.py。

## 当前版本边界

- 前10个完整回合=双方各10步=20个半回合；不是双方总共走10步。
- 不依赖Stockfish，不包括深度学习，不比较不同算法。
- 仅用截至第20个半回合的信息。第21步之后的实际棋谱不参与特征。
- 所有实验行均为原时间分区中的同一队列；两组模型输入有6列或46列。
- 类别顺序固定black、draw、white；整数映射0、1、2。

## 数据成员

查看tables/plan_a20/feature_dictionary.csv和row_decisions.csv。
完整审计表含winner、完整棋谱、结束方式等，只有审计和特征提取时使用，不能直接训练。
需要重建时先运行verify_snapshot.py，修改实验请使用新分支，保留源数据和manifest。
rebuild_data.py先在临时目录重做原清洗，确认清洗数据和manifest哈希匹配后才复制，再生成本版输入；不会把旧赛前版文件混入共享数据目录。

## 建模成员

使用src.plan_a20_interface.load_split；feature_set可为pregame或combined。
每个训练折使用新的未拟合Pipeline。不要先对全部数据做缩放再CV。
当前参数是验证阶段固定的200树、深度16、叶节点至少10样本、max_features=sqrt、class_weight=balanced、random_state=42。
先在训练内时间CV和固定验证集做实验；正式运行使用modelling_rf/local_自定义名称。
旧的已完成运行目录不允许直接重做tune/finalize，这是为防止覆盖结果。

## 评估成员

同一批测试对局、相同顺序、相同类别顺序，输出Macro-F1、Accuracy、Balanced Accuracy、分类别Precision/Recall/F1、混淆矩阵及Log Loss。
本版已正式评分；不要靠反复查看测试集来选参数、特征或时点。后续同一测试期的修改只能按探索性结果描述。
最终模型在train+validation共14,485场重新拟合；train_only模型用于原验证解释。不要用最终模型在验证集的成绩冒充独立验证。
验证集置换重要性来自train_only模型，不能解释为因果。

## 查找已有文件

models/rf_combined_final.joblib：本版主模型，46列输入。
models/rf_pregame_final.joblib：同批对局上的6列对照。
models/*_train_only.joblib：用于原开发集验证及解释。
上述models均位于modelling_rf/run_20261025_20ply_01/。
完整环境安装后，可用根目录predict_opening.py对一场新棋谱预测，或在Notebook中读取已保存的结果。

## 协作建议

在各自分支工作，以PR交接，写清改变的特征/参数、验证依据、是否触碰测试集。
只提交有用的代码、表格和结果，不提交.venv、缓存或临时运行；local_*实验目录被.gitignore排除，需要分享时有意识地选择文件。
数据CSV明确禁止Git换行转换，以保证Mac/Windows克隆后的来源哈希一致。
