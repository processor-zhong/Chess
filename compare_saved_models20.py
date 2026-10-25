"""Combine saved development results, without training or reading test files."""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.chess_data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-neural', action='store_true',
                        help='Optional archived experiments; excluded from submission by default')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    base = root / 'modelling_baselines'
    out = base / ('comparison_neural_archive' if args.include_neural else 'comparison_20ply')
    out.mkdir(exist_ok=True)
    original = base / 'run_20261025_20ply_01/tables/validation_metrics.csv'
    tree = base / 'run_20261025_tree_01/tables/validation_metrics.csv'
    a, b = pd.read_csv(original), pd.read_csv(tree).assign(family='decision_tree')
    frames = [a, b]
    inputs = [original, tree]
    mlp_run = base / 'run_20261025_mlp_01'
    include_mlp = args.include_neural and (mlp_run / 'artifacts/development_complete.json').exists()
    if include_mlp:
        mlp_path = mlp_run / 'tables/validation_metrics.csv'
        frames.append(pd.read_csv(mlp_path).query('seed == 42'))
        inputs += [mlp_path, mlp_run / 'tables/seed_stability.csv']
    dnn_run = base / 'run_20261025_dnn_01'
    include_dnn = args.include_neural and (dnn_run / 'artifacts/development_complete.json').exists()
    if include_dnn:
        dnn_path = dnn_run / 'tables/validation_metrics.csv'
        frames.append(pd.read_csv(dnn_path).query('seed == 42'))
        inputs += [dnn_path, dnn_run / 'tables/seed_stability.csv']
    grouped_run = base / 'run_20261025_grouped_01'
    include_grouped = args.include_neural and (grouped_run / 'artifacts/development_complete.json').exists()
    if include_grouped:
        inputs += [grouped_run / 'tables/validation_metrics.csv', grouped_run / 'tables/seed_stability.csv']
    sequence_run = base / 'run_20261025_sequence_01'
    include_sequence = args.include_neural and (sequence_run / 'artifacts/development_complete.json').exists()
    if include_sequence:
        inputs += [sequence_run / 'tables/validation_metrics.csv', sequence_run / 'tables/seed_stability.csv']
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out / 'all_initial_tuned_metrics.csv', index=False)
    selected = combined[(combined.stage == 'tuned') | (combined.family == 'dummy')]
    selected.to_csv(out / 'cv_selected_metrics.csv', index=False)
    names = dict(dummy='多数类基线', logistic='逻辑回归', gaussian_nb='高斯朴素贝叶斯',
                 decision_tree='单棵决策树', random_forest='随机森林', gradient_boosting='梯度提升')
    english = ['Majority', 'Logistic', 'GaussianNB', 'Decision tree', 'Random forest', 'HistGB']
    if include_mlp:
        names['mlp'] = '多层感知机MLP'
        english.append('MLP')
    if include_dnn:
        names['dnn'] = '深层全连接DNN'
        english.append('DNN')
    families = list(names)
    positions = np.arange(len(families))
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), layout='constrained')
    for row, fs in enumerate(['pregame', 'combined']):
        for ax, metric, title in zip(axes[row], ['accuracy', 'macro_f1', 'draw_recall'],
                                     ['Accuracy', 'Macro-F1', 'Draw recall']):
            for stage, offset, color in [('initial', -.18, '#52758C'), ('tuned', .18, '#C77A3D')]:
                values = combined[(combined.feature_set == fs) & (combined.stage == stage)]
                values = values.set_index('family').reindex(families)[metric]
                bars = ax.bar(positions + offset, values, .34, label=stage.title(), color=color)
                ax.bar_label(bars, labels=['' if pd.isna(v) else f'{v:.3f}' for v in values],
                             fontsize=7, padding=3, rotation=90)
            ax.set(xticks=positions, xticklabels=english, ylim=(0, 1), title=f'{fs}: validation {title}')
            ax.tick_params(axis='x', labelrotation=30)
            ax.legend(fontsize=8)
    fig.suptitle('Same validation games at 20 plies; tuned models selected using training temporal CV')
    fig.savefig(out / 'all_models_comparison.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    lines = ['# 全部基础模型的统一开发比较', '',
             '相同20半回合观察时点、训练11,936局、验证2,549局，黑胜／和棋／白胜三分类。',
             '除多数类基线外，主表使用训练时间CV选定参数的模型，不按验证结果挑选初始或调参版本。',
             '本脚本只合并保存结果，不训练、不读取测试文件。', '',
             '| 输入 | 模型 | 验证Accuracy | Macro-F1 | Balanced Accuracy | 和棋Precision | 和棋Recall | Log Loss |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for fs in ['pregame', 'combined']:
        d = selected[selected.feature_set == fs].set_index('family').loc[families]
        for family, r in d.iterrows():
            lines.append(f'| {fs} | {names[family]} | {r.accuracy:.2%} | {r.macro_f1:.4f} | '
                         f'{r.balanced_accuracy:.2%} | {r.draw_precision:.2%} | {r.draw_recall:.2%} | {r.log_loss:.4f} |')
    current = selected[selected.feature_set == 'combined']
    best_accuracy = current.loc[current.accuracy.idxmax()]
    best_f1 = current.loc[current.macro_f1.idxmax()]
    lines += ['', '## 对当前46列结果的解释', '',
              f'- 准确率最高：{names[best_accuracy.family]} {best_accuracy.accuracy:.2%}；差异只作描述，没有显著性结论。',
              f'- Macro-F1最高：{names[best_f1.family]} {best_f1.macro_f1:.4f}，对应Accuracy={best_f1.accuracy:.2%}。',
              '- 和棋：RF识别28/120场，Precision仅9.21%；NB识别3/120场；其余结果见表，不能仅依据Recall判断。',
              '- 单棵树：从默认38层、训练99.99%／验证52.65%，约束到3层后训练63.56%／验证60.22%；过拟合缓解，但当前综合结果未超过LR。',
              '- 概率损失越低越好，需与Accuracy分别判断；Dummy的0/1概率不能代表合理的概率参照。',
              '- 用LR作准确率／概率基线、用加权RF展示少数类识别取舍；不能只选对某模型有利的指标。']
    if include_mlp:
        lines += ['', '## MLP随机初始化检查', '',
                  '主表仅用预先固定seed42。下面为相同选定参数在seed42/7/2026的全部结果汇总，未挑选最佳种子。', '',
                  '| 输入 | Accuracy均值 | 三次初始化标准差 | 最小Accuracy | 最大Accuracy | Macro-F1均值 |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in pd.read_csv(mlp_run / 'tables/seed_stability.csv').itertuples():
            lines.append(f'| {r.feature_set} | {r.accuracy_mean:.2%} | {r.accuracy_sd:.2%} | '
                         f'{r.accuracy_min:.2%} | {r.accuracy_max:.2%} | {r.macro_f1_mean:.4f} |')
    if include_dnn:
        lines += ['', '## DNN随机初始化检查', '',
                  '主表固定seed42；结构、内部早停轮数和全部模型在打开外部验证标签前冻结。',
                  '与上轮MLP同时改变了深度、正则化及早停方案，差异不能只归因于加深层数。', '',
                  '| 输入 | Accuracy均值 | 三次初始化标准差 | 最小Accuracy | 最大Accuracy | Macro-F1均值 |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in pd.read_csv(dnn_run / 'tables/seed_stability.csv').itertuples():
            lines.append(f'| {r.feature_set} | {r.accuracy_mean:.2%} | {r.accuracy_sd:.2%} | '
                         f'{r.accuracy_min:.2%} | {r.accuracy_max:.2%} | {r.macro_f1_mean:.4f} |')
    if include_grouped:
        lines += ['', '## 分组编码结构补充（仅46列）', '',
                  '同批对局，赛前6列和开局40列分路输入；不是新增走法序列。主种子固定42。', '',
                  '| 结构 | Accuracy | Macro-F1 | 和棋Recall | 参数量 |',
                  '|---|---:|---:|---:|---:|']
        grouped = pd.read_csv(grouped_run / 'tables/validation_metrics.csv')
        group_names = {'flat': '直接拼接对照', 'separate': '分组独立编码', 'shared': '分组共享编码'}
        for r in grouped[grouped.seed == 42].itertuples():
            lines.append(f'| {group_names[r.architecture]} | {r.accuracy:.2%} | {r.macro_f1:.4f} | '
                         f'{r.draw_recall:.2%} | {r.parameter_count} |')
        lines += ['', '共享版本对直接拼接主种子仅提高约0.31个百分点；未超过逻辑回归。'
                  '没有显著性结论；详见docs/GROUPED_FEATURES_ZH.md及全部种子记录。']
    if include_sequence:
        lines += ['', '## 前20半回合逐步共享编码（新增序列表示）', '',
                  '相同对局与赛前6列，但开局表示改为20步×8类别字段；主种子固定42。', '',
                  '| 汇聚方式 | Accuracy | Macro-F1 | 和棋Recall | 参数量 |',
                  '|---|---:|---:|---:|---:|']
        sequence_metrics = pd.read_csv(sequence_run / 'tables/validation_metrics.csv')
        for r in sequence_metrics[sequence_metrics.seed == 42].itertuples():
            lines.append(f'| {r.aggregation} | {r.accuracy:.2%} | {r.macro_f1:.4f} | '
                         f'{r.draw_recall:.2%} | {r.parameter_count} |')
        lines += ['', '每步共用编码器；mean不保留精确顺序，GRU按时间汇聚，但GRU参数更多。'
                  '与40列汇总版本相比也改变了信息表示，不能当作纯结构消融。全部种子见序列实验结果目录。']
    lines += ['',
              '## 边界', '',
              '- 各模型搜索量为4到20组不等，均为有限搜索，不能声称对比了算法各自的最优能力。',
              '- 调参目标是训练CV Macro-F1，验证上没有提升也如实保留。',
              '- 验证和历史测试已被此前探索查看，以上是开发比较，不能视为新的盲测。',
              '- 不按参考项目的精确开局分组删除罕见前缀，不使用重复对局泄漏换取高分。']
    (out / 'COMPARISON_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    sources = {'input_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in inputs},
               'script_sha256': sha256(Path(__file__)), 'model_training': False, 'test_loaded': False}
    (out / 'provenance.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    assert len(selected) == 2 * len(families) and not selected.duplicated(['feature_set', 'family']).any()
    print(selected[selected.feature_set.eq('combined')][
        ['family', 'accuracy', 'macro_f1', 'draw_precision', 'draw_recall']].to_string(index=False))
    print('Saved combined comparison; no training or test access.')


if __name__ == '__main__':
    main()
