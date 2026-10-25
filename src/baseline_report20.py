"""Read-only reporting and verification for development model comparisons."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, log_loss, roc_auc_score, roc_curve)

from .chess_data import LABELS, sha256

ENGLISH = dict(dummy='Majority', logistic='Logistic', gaussian_nb='GaussianNB',
               random_forest='Random forest', gradient_boosting='HistGB')


def draw_figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    metrics = pd.read_csv(out / 'tables/validation_metrics.csv')
    families = list(ENGLISH)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.savefig(out / f'figures/{name}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for row, fs in enumerate(['pregame', 'combined']):
        d = metrics[metrics.feature_set == fs]
        for ax, metric, title in zip(axes[row], ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
            for stage, offset, color in [('initial', -.18, '#52758C'), ('tuned', .18, '#C77A3D')]:
                subset = d[d.stage == stage].set_index('family').reindex(families)
                bars = ax.bar(np.arange(len(families)) + offset, subset[metric], .34,
                              color=color, label=stage.title())
                ax.bar_label(bars, labels=['' if pd.isna(v) else f'{v:.3f}' for v in subset[metric]],
                             fontsize=8, padding=2)
            ax.set(xticks=range(len(families)), xticklabels=[ENGLISH[f] for f in families],
                   ylim=(0, 1), title=f'{fs}: validation {title}')
            ax.tick_params(axis='x', labelrotation=15)
            ax.legend()
    fig.suptitle('Same games and chronological folds; parameters selected by training CV Macro-F1')
    save(fig, '01_validation_comparison')

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), layout='constrained')
    for row, stage in enumerate(['initial', 'tuned']):
        for col, family in enumerate(families):
            actual_stage = 'initial' if family == 'dummy' else stage
            slug = f'combined_{family}_{actual_stage}'
            cm = pd.read_csv(out / f'tables/{slug}_confusion.csv', index_col=0).to_numpy()
            normalized = cm / cm.sum(axis=1, keepdims=True)
            ax = axes[row, col]
            ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{cm[i,j]}\n{normalized[i,j]:.0%}', ha='center', va='center',
                            fontsize=8, color='white' if normalized[i,j] > .55 else 'black')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
                   title=f'{ENGLISH[family]} ({actual_stage})', xlabel='Predicted', ylabel='Actual')
    fig.suptitle('46-feature validation confusion matrices; count and within-class recall')
    save(fig, '02_combined_confusion')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    learned = families[1:]
    for ax, fs in zip(axes, ['pregame', 'combined']):
        d = metrics[(metrics.feature_set == fs) & (metrics.stage == 'tuned')].set_index('family').loc[learned]
        positions = np.arange(len(learned))
        ax.bar(positions-.18, d.train_macro_f1, .34, color='#52758C', label='Train')
        ax.bar(positions+.18, d.macro_f1, .34, color='#C77A3D', label='Validation')
        ax.plot(positions, d.cv_macro_f1, 'ko', markersize=4, label='Mean temporal CV')
        ax.set(xticks=positions, xticklabels=[ENGLISH[f] for f in learned],
               ylim=(0, 1), ylabel='Macro-F1', title=f'{fs}: CV-selected models')
        ax.tick_params(axis='x', labelrotation=15)
        ax.legend()
    save(fig, '03_training_validation_gap')

    cv = pd.read_csv(out / 'tables/cv_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, family in zip(axes.flat, learned):
        for fs, color in [('pregame', '#52758C'), ('combined', '#C77A3D')]:
            d = cv[(cv.family == family) & (cv.feature_set == fs) & (cv.candidate >= 0)]
            ax.errorbar(d.candidate, d.cv_macro_f1, yerr=d.cv_macro_f1_std,
                        marker='o', markersize=4, capsize=2, label=fs, color=color)
        ax.set(title=ENGLISH[family], xlabel='Candidate ID (parameters in cv_summary.csv)',
               ylabel='Mean training CV Macro-F1')
        ax.legend()
    fig.suptitle('Error bars: fold sample SD; not confidence intervals')
    save(fig, '04_search_cv_scores')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    for ax, label in zip(axes, LABELS):
        for family in learned:
            d = pd.read_csv(out / f'tables/combined_{family}_tuned_predictions.csv')
            y = d.actual.eq(label)
            p = d[f'prob_{label}']
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=f'{ENGLISH[family]} {roc_auc_score(y,p):.3f}')
        ax.plot([0, 1], [0, 1], ':', color='gray')
        ax.set(title=f'{label} vs rest', xlabel='False positive rate', ylabel='Recall',
               xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle('46-feature validation ROC; tuned models')
    save(fig, '05_combined_roc')


def write_report(out):
    from .baseline_models20 import NAMES
    out = Path(out)
    d = pd.read_csv(out / 'tables/validation_metrics.csv')
    frozen = json.loads((out / 'artifacts/frozen_config.json').read_text(encoding='utf-8'))
    lines = ['# 基础模型对比与有限调参：20个半回合', '',
             '使用同一训练集11,936局与验证集2,549局，保留黑胜、和棋、白胜三分类。',
             '赛前6列与赛前＋开局46列分别比较；本轮没有读取测试集或旧测试预测。', '',
             '## 验证结果', '',
             '| 输入 | 模型 | 阶段 | Accuracy | Macro-F1 | 和棋Precision | 和棋Recall | Log Loss |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in d.itertuples():
        stage = '初始' if r.stage == 'initial' else 'CV调参后'
        lines.append(f'| {r.feature_set} | {NAMES[r.family]} | {stage} | {r.accuracy:.2%} | '
                     f'{r.macro_f1:.4f} | {r.draw_precision:.2%} | {r.draw_recall:.2%} | {r.log_loss:.4f} |')
    lines += ['', '## 选定参数', '', '每个模型、每套输入独立选择，全部在打开验证标签之前冻结。', '']
    for c in frozen['selected']:
        lines.append(f'- {c["feature_set"]} / {NAMES[c["family"]]}：`{json.dumps(c["parameters"], ensure_ascii=False)}`')
    lines += ['', '## 如何解释', '',
              '- 主要选参指标是训练内三折时间CV的平均Macro-F1；并列时用CV Log Loss和候选序号打破平局。验证结果不用于重新选参。',
              '- CV选参不保证验证准确率、和棋Recall或Macro-F1全部上升，调参失败也保留并报告。',
              '- 初始设置沿用仓库starter，不能全部称为sklearn默认值。随机森林初始300树、叶样本5；梯度提升初始150轮、15叶、关闭随机内部早停。',
              '- 逻辑回归与高斯朴素贝叶斯在每折内标准化；树模型不标准化。全部模型在每折内拟合插补和常量特征移除。',
              '- GaussianNB的均匀先验与class_weight不同。标准化不会使数据自动服从高斯分布；局面计数和派生变量可能不满足条件独立假设。',
              '- 搜索量：LR 8组、GaussianNB 8组、RF 12组、HistGB 8组，每组每个信息集3折。搜索预算有差别，不能声称找到了各算法全局最优参数。',
              '- 最终模型文件只在训练集拟合，用于验证解释与复现；没有在train+validation重训，也没有新的测试成绩。',
              '- 原仓库RF为两套特征选择共享参数；本轮为每个模型和信息集分别按训练CV选择。它们属于不同选参协议，不能把分数差全部归因为算法。',
              '- 验证集经过此前多轮探索，历史测试也已被查看；本轮是开发比较，不能宣称全新盲测结论或显著提升。',
              '- 均匀随机准确率33%不足以作为唯一参照；表中提供实际训练多数类模型。', '',
              '## 文件', '',
              '- tables/validation_metrics.csv：总体指标、训练分数、CV分数、参数和耗时。',
              '- tables/tuning_changes.csv：同模型初始到调参后的变化；Log Loss变化越负越好。',
              '- tables/cv_summary.csv、cv_fold_metrics.csv：全部候选的训练CV记录。',
              '- tables/*_predictions.csv、*_confusion.csv：逐局三类概率与混淆矩阵。',
              '- figures/：模型比较、混淆矩阵、拟合差距、CV搜索和ROC图。',
              '- artifacts/：流程、源文件哈希、冻结参数和核对记录。',
              '- models/：每个初始／调参模型的训练集Pipeline。', '',
              '本文件由实验脚本生成；复现请使用新的输出目录，不能覆盖已有结果。']
    (out / 'RESULTS_ZH.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def verify_results(root, out, reload_models=False):
    from .plan_a20_interface import load_split
    from .baseline_models20 import select_candidate, parameter_key
    root, out = Path(root), Path(out)
    checks = []

    def check(ok, name):
        if not bool(ok):
            raise AssertionError(name)
        checks.append(name)

    d = pd.read_csv(out / 'tables/validation_metrics.csv')
    protocol = json.loads((out / 'artifacts/protocol.json').read_text(encoding='utf-8'))
    frozen = json.loads((out / 'artifacts/frozen_config.json').read_text(encoding='utf-8'))
    cv = pd.read_csv(out / 'tables/cv_summary.csv')
    check(sha256(out / 'artifacts/protocol.json') == frozen['protocol_sha256'], 'Frozen protocol hash')
    check(not frozen['test_loaded'] and not frozen['validation_labels_loaded_during_selection'],
          'Recorded selection used no test or validation labels')
    for path, expected in protocol['source_sha256'].items():
        check(sha256(root / path) == expected, f'Source hash: {path}')
    for choice in frozen['selected']:
        block = cv[(cv.feature_set == choice['feature_set']) & (cv.family == choice['family'])
                   & (cv.candidate >= 0)]
        check(select_candidate(block[['candidate', 'cv_macro_f1', 'cv_log_loss']]) == choice['candidate'],
              f'CV choice reproduced: {choice["feature_set"]}/{choice["family"]}')
    check(len(d) == 18 and d.slug.is_unique, '18 unique model/feature/stage evaluations')
    for fs in ['pregame', 'combined']:
        X, y, meta = load_split(root, 'validation', feature_set=fs)
        for r in d[d.feature_set == fs].itertuples():
            p = pd.read_csv(out / f'tables/{r.slug}_predictions.csv')
            probability = p[[f'prob_{c}' for c in LABELS]].to_numpy()
            check(p.id.equals(meta.id) and p.actual.equals(y), f'Row and label alignment: {r.slug}')
            check(np.isfinite(probability).all() and (probability >= 0).all()
                  and np.allclose(probability.sum(axis=1), 1), f'Probability validity: {r.slug}')
            # Probability argmax and estimator predict must agree (including ties).
            expected_prediction = np.asarray(LABELS)[probability.argmax(axis=1)]
            check(np.array_equal(p.predicted, expected_prediction), f'Argmax: {r.slug}')
            expected = {'accuracy': accuracy_score(y, p.predicted),
                        'macro_f1': f1_score(y, p.predicted, labels=LABELS, average='macro', zero_division=0),
                        'balanced_accuracy': balanced_accuracy_score(y, p.predicted),
                        'log_loss': log_loss(y, probability, labels=LABELS),
                        'macro_auc': roc_auc_score(y, probability, labels=LABELS, multi_class='ovr', average='macro')}
            for name, value in expected.items():
                check(np.isclose(value, getattr(r, name), rtol=1e-10, atol=1e-10), f'{name}: {r.slug}')
            cm = pd.read_csv(out / f'tables/{r.slug}_confusion.csv', index_col=0).to_numpy()
            check(np.array_equal(cm, confusion_matrix(y, p.predicted, labels=LABELS)), f'Confusion: {r.slug}')
            check(np.isclose(cm[1, 1] / cm[1].sum(), r.draw_recall), f'Draw recall: {r.slug}')
            if reload_models:
                model = joblib.load(out / f'models/{r.slug}.joblib')
                order = [list(model.classes_).index(c) for c in LABELS]
                check(np.array_equal(model.predict(X), p.predicted)
                      and np.allclose(model.predict_proba(X)[:, order], probability), f'Model reload: {r.slug}')
                params = json.loads(r.parameters)
                check(parameter_key(r.family, params) == json.dumps(model.named_steps['model'].get_params(),
                                                                    sort_keys=True, default=str),
                      f'Exported model parameters: {r.slug}')
    return {'status': 'passed', 'check_count': len(checks), 'checks': checks,
            'test_loaded': False, 'reloaded_models': reload_models}
