"""Add a single-tree baseline and chronological-CV regularization comparison.

Uses only the same frozen development data as the other baseline models.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

from src.baseline_models20 import (now, predict, reserve_output, save_json,
                                   scores, select_candidate)
from src.chess_data import LABELS, sha256
from src.plan_a20_interface import chronological_cv, load_split

DEFAULT_OUTPUT = 'modelling_baselines/run_20261025_tree_01'


def build(params):
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('nonconstant', VarianceThreshold()),
                     ('model', DecisionTreeClassifier(random_state=42, **params))])


def run(root, output):
    root = Path(root)
    out = reserve_output(root, output)
    data = {fs: load_split(root, 'train', feature_set=fs) for fs in ['pregame', 'combined']}
    folds = list(chronological_cv(data['combined'][2]))
    grid = [dict(max_depth=d, min_samples_leaf=l, class_weight=w)
            for d in [3, 5, 8, 12, None] for l in [5, 20] for w in [None, 'balanced']]
    sources = [root / f'data/plan_a20/splits/{p}_{s}.csv' for p in ['X', 'y', 'meta']
               for s in ['train', 'validation']]
    sources += [Path(__file__), root / 'src/baseline_models20.py', root / 'src/plan_a20_interface.py',
                root / 'src/model_interface.py', root / 'artifacts/plan_a20/feature_contract.json']
    protocol = {
        'created_utc': now(), 'algorithm': 'DecisionTreeClassifier', 'prediction_ply': 20,
        'labels': LABELS, 'seed': 42, 'initial': 'sklearn defaults except random_state=42',
        'grid': grid, 'selection': 'Per feature set: mean training temporal CV Macro-F1, '
        'then lower CV Log Loss, then candidate ID. Frozen before validation labels loaded.',
        'cv': 'Three expanding chronological folds with whole UTC days and 24-hour gap; fresh Pipeline',
        'preprocessing': 'Fold-fitted median imputation and constant removal, no scaling',
        'test_loaded': False, 'python': platform.python_version(),
        'sklearn': importlib.metadata.version('scikit-learn'),
        'source_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in sources},
    }
    save_json(protocol, out / 'artifacts/protocol.json')
    cv_rows, summary, chosen = [], [], []
    for fs, (X, y, _) in data.items():
        for ci, params in [(-1, {}), *enumerate(grid)]:
            fold_scores = []
            for fold, (tr, va) in enumerate(folds, 1):
                model = build(params).fit(X.iloc[tr], y.iloc[tr])
                prediction, probability = predict(model, X.iloc[va])
                m, _, _ = scores(y.iloc[va], prediction, probability)
                fold_scores.append(m)
                cv_rows.append({'feature_set': fs, 'candidate': ci, 'fold': fold, **m})
            summary.append({'feature_set': fs, 'candidate': ci, 'parameters': json.dumps(params),
                            'cv_macro_f1': np.mean([s['macro_f1'] for s in fold_scores]),
                            'cv_log_loss': np.mean([s['log_loss'] for s in fold_scores]),
                            'cv_accuracy': np.mean([s['accuracy'] for s in fold_scores])})
        table = pd.DataFrame([s for s in summary if s['feature_set'] == fs and s['candidate'] >= 0])
        ci = select_candidate(table[['candidate', 'cv_macro_f1', 'cv_log_loss']])
        chosen.append({'feature_set': fs, 'candidate': ci, 'parameters': grid[ci]})
        print(f'CV complete: {fs}, selected {grid[ci]}', flush=True)
    pd.DataFrame(cv_rows).to_csv(out / 'tables/cv_fold_metrics.csv', index=False)
    pd.DataFrame(summary).to_csv(out / 'tables/cv_summary.csv', index=False)
    save_json({'frozen_utc': now(), 'selected': chosen, 'validation_labels_loaded': False,
               'test_loaded': False, 'protocol_sha256': sha256(out / 'artifacts/protocol.json')},
              out / 'artifacts/frozen_config.json')
    rows, perclass, checks = [], [], []
    matrices = {}
    for choice in chosen:
        fs = choice['feature_set']
        X, y, mt = data[fs]
        Xv, yv, mv = load_split(root, 'validation', feature_set=fs)
        assert not set(mt.id) & set(mv.id)
        for stage, ci, params in [('initial', -1, {}), ('tuned', choice['candidate'], choice['parameters'])]:
            slug = f'{fs}_{stage}'
            tick = time.perf_counter()
            model = build(params).fit(X, y)
            elapsed = time.perf_counter() - tick
            prediction, probability = predict(model, Xv)
            m, report, cm = scores(yv, prediction, probability)
            tp, tprob = predict(model, X)
            tm, _, _ = scores(y, tp, tprob)
            cv = next(s for s in summary if s['feature_set'] == fs and s['candidate'] == ci)
            tree = model.named_steps['model']
            rows.append({'feature_set': fs, 'stage': stage, 'parameters': json.dumps(params),
                         'train_accuracy': tm['accuracy'], 'train_macro_f1': tm['macro_f1'],
                         'actual_depth': tree.get_depth(), 'leaf_count': tree.get_n_leaves(),
                         'fit_seconds': elapsed, 'cv_macro_f1': cv['cv_macro_f1'], **m})
            for label in LABELS:
                perclass.append({'feature_set': fs, 'stage': stage, 'class': label, **report[label]})
            predictions = mv[['id', 'created_at_utc']].copy()
            predictions['actual'] = yv.to_numpy()
            predictions['predicted'] = prediction
            for i, label in enumerate(LABELS): predictions[f'prob_{label}'] = probability[:, i]
            predictions.to_csv(out / f'tables/{slug}_predictions.csv', index=False)
            pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(out / f'tables/{slug}_confusion.csv')
            matrices[slug] = cm
            joblib.dump(model, out / f'models/{slug}.joblib', compress=3)
            restored = joblib.load(out / f'models/{slug}.joblib')
            rp, rprob = predict(restored, Xv)
            assert np.array_equal(rp, prediction) and np.allclose(rprob, probability)
            checks.append(f'{slug}: model reload reproduces validation predictions/probabilities')
            assert int(np.trace(cm)) == int(np.count_nonzero(prediction == yv.to_numpy()))
            checks.append(f'{slug}: confusion matches correct prediction count')
            feature_names = np.asarray(X.columns)[model.named_steps['nonconstant'].get_support()].tolist()
            (out / f'tables/{slug}_top_tree.txt').write_text(
                export_text(tree, feature_names=feature_names, max_depth=3), encoding='utf-8')
            print(f'{slug}: train={tm["accuracy"]:.4f}, validation={m["accuracy"]:.4f}, '
                  f'Macro-F1={m["macro_f1"]:.4f}, depth={tree.get_depth()}', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'tables/validation_metrics.csv', index=False)
    pd.DataFrame(perclass).to_csv(out / 'tables/validation_per_class.csv', index=False)
    for relative, expected in protocol['source_sha256'].items():
        assert sha256(root / relative) == expected
        checks.append('Unchanged source: ' + relative)
    save_json({'status': 'passed', 'check_count': len(checks), 'checks': checks,
               'test_loaded': False}, out / 'artifacts/result_verification.json')
    report = ['# 单棵决策树补充实验', '',
              '相同20半回合预测时点、原始三分类、相同训练11,936局和验证2,549局。未读取测试集。',
              '初始设置仅固定random_state=42；调参20组，每套特征训练内3折时间CV。按Macro-F1选参，冻结后评价验证集。', '',
              '| 输入 | 阶段 | 实际深度 | 叶节点数 | 训练Accuracy | 验证Accuracy | 验证Macro-F1 | 和棋Precision | 和棋Recall |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result.itertuples():
        report.append(f'| {r.feature_set} | {r.stage} | {r.actual_depth} | {r.leaf_count} | '
                      f'{r.train_accuracy:.2%} | {r.accuracy:.2%} | {r.macro_f1:.4f} | '
                      f'{r.draw_precision:.2%} | {r.draw_recall:.2%} |')
    report += ['', '## 选定参数', '']
    for c in chosen: report.append(f'- {c["feature_set"]}: `{json.dumps(c["parameters"])}`')
    report += ['', '## 解释边界', '',
               '- 单棵树初始设置不限制深度，小叶节点可能记住训练数据。训练分数只作拟合诊断，不用于选参。',
               '- 调参目的为训练时间CV Macro-F1；不保证验证准确率或每个类别同时改善。',
               '- 本模型是统一的特征决策树，不是参考项目按完整开局字符串分组、排除低频开局的方法。',
               '- 本轮搜索20组，前一轮其他模型8或12组，均为有限搜索，不代表全局最优或相同搜索预算。',
               '- 历史测试和验证经过此前探索，本次仍为开发比较；没有新的盲测成绩。',
               '- tables/*_top_tree.txt仅展示树顶部；图表和逐局概率保存在同目录。']
    (out / 'RESULTS_ZH.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    figures(out, result, matrices)
    save_json({'completed_utc': now(), 'test_loaded': False,
               'metrics_sha256': sha256(out / 'tables/validation_metrics.csv')},
              out / 'artifacts/development_complete.json')
    save_json([{'path': p.relative_to(out).as_posix(), 'sha256': sha256(p)}
               for p in sorted(out.rglob('*')) if p.is_file()], out / 'artifacts/output_hashes.json')


def figures(out, result, matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout='constrained')
    for row, fs in enumerate(['pregame', 'combined']):
        for col, stage in enumerate(['initial', 'tuned']):
            cm = matrices[f'{fs}_{stage}']; fractions = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[row, col]; ax.imshow(fractions, cmap='Blues', vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{cm[i,j]}\n{fractions[i,j]:.1%}', ha='center', va='center',
                            color='white' if fractions[i,j] > .55 else 'black')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
                   xlabel='Predicted', ylabel='Actual', title=f'{fs}: {stage}')
    fig.suptitle('Single decision tree: fixed validation confusion matrices')
    fig.savefig(out / 'figures/confusion_matrices.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
    for ax, fs in zip(axes, ['pregame', 'combined']):
        d = result[result.feature_set == fs]
        ax.bar(np.arange(2)-.18, d.train_accuracy, .34, label='Train', color='#52758C')
        ax.bar(np.arange(2)+.18, d.accuracy, .34, label='Validation', color='#C77A3D')
        ax.set(xticks=range(2), xticklabels=d.stage, ylim=(0, 1.05), ylabel='Accuracy', title=fs)
        ax.legend()
    fig.savefig(out / 'figures/training_validation_gap.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, args.output)
