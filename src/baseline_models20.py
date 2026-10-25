"""Development-only model comparison at the frozen 20-ply observation point.

All parameter choices use training-only chronological CV. Validation is scored
after those choices are frozen. This module has no final-test evaluation stage.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, log_loss, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .chess_data import LABELS, SEED, sha256
from .plan_a20_interface import load_split, make_model, chronological_cv

FAMILIES = ['dummy', 'logistic', 'gaussian_nb', 'random_forest', 'gradient_boosting']
NAMES = dict(dummy='多数类基线', logistic='逻辑回归', gaussian_nb='高斯朴素贝叶斯',
             random_forest='随机森林', gradient_boosting='直方图梯度提升')
FEATURE_SETS = ['pregame', 'combined']
DEFAULT_OUTPUT = 'modelling_baselines/run_20261025_20ply_01'


def save_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=str) + '\n', encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def specifications(family):
    """Return initial configuration and small, explicitly enumerated search."""
    if family == 'dummy':
        return {}, []
    if family == 'logistic':
        return {}, [dict(C=c, class_weight=w) for c in [.01, .1, 1., 10.]
                    for w in [None, 'balanced']]
    if family == 'gaussian_nb':
        return {}, [dict(var_smoothing=v, priors=prior)
                    for v in [1e-9, 1e-6, 1e-3, .1]
                    for prior in [None, [1/3, 1/3, 1/3]]]
    if family == 'random_forest':
        return {}, [dict(n_estimators=200, max_depth=d, min_samples_leaf=l,
                         max_features='sqrt', class_weight=w)
                    for d in [8, 16, None] for l in [3, 10] for w in [None, 'balanced']]
    if family == 'gradient_boosting':
        return {}, [dict(max_iter=150, learning_rate=.1, max_leaf_nodes=l,
                         l2_regularization=reg, class_weight=w)
                    for l in [7, 15] for reg in [0., 10.] for w in [None, 'balanced']]
    raise ValueError(f'Unknown model family: {family}')


def build_model(family, parameters=None):
    parameters = dict(parameters or {})
    if family == 'gaussian_nb':
        return Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('nonconstant', VarianceThreshold()),
                         ('scaler', StandardScaler()),
                         ('model', GaussianNB(**parameters))])
    model = make_model(family, **parameters)
    if family == 'random_forest':
        model.set_params(model__n_jobs=4)
    return model


def parameter_key(family, parameters):
    return json.dumps(build_model(family, parameters).named_steps['model'].get_params(),
                      sort_keys=True, default=str)


def select_candidate(summary):
    """Only training-CV scores are accepted; validation never enters selection."""
    required = ['candidate', 'cv_macro_f1', 'cv_log_loss']
    if set(summary.columns) != set(required) or summary.empty:
        raise ValueError('Selection requires exactly training CV candidate/F1/log-loss columns')
    if not np.isfinite(summary[required].to_numpy(dtype=float)).all():
        raise ValueError('Candidate scores must be finite')
    ranked = summary.sort_values(['cv_macro_f1', 'cv_log_loss', 'candidate'],
                                 ascending=[False, True, True])
    return int(ranked.iloc[0].candidate)


def reserve_output(root, relative):
    root = Path(root).resolve()
    out = (root / relative).resolve()
    if out == root or root not in out.parents:
        raise ValueError('Output must be a new directory inside the checkout')
    if out.exists():
        raise FileExistsError('Output already exists; choose a new --output directory')
    out.mkdir(parents=True)
    for name in ['tables', 'models', 'figures', 'artifacts']:
        (out / name).mkdir()
    return out


def scores(y, prediction, probability):
    if probability.shape != (len(y), len(LABELS)):
        raise ValueError('Unexpected probability shape')
    if (not np.isfinite(probability).all() or (probability < 0).any()
            or not np.allclose(probability.sum(axis=1), 1)):
        raise ValueError('Invalid probabilities')
    report = classification_report(y, prediction, labels=LABELS,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y, prediction, labels=LABELS)
    result = {
        'accuracy': accuracy_score(y, prediction),
        'macro_f1': f1_score(y, prediction, labels=LABELS, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y, prediction),
        'draw_precision': report['draw']['precision'],
        'draw_recall': report['draw']['recall'],
        'draw_f1': report['draw']['f1-score'],
        'log_loss': log_loss(y, probability, labels=LABELS),
        'macro_auc': roc_auc_score(y, probability, labels=LABELS,
                                   multi_class='ovr', average='macro'),
    }
    return result, report, cm


def predict(model, X):
    order = [list(model.classes_).index(label) for label in LABELS]
    return model.predict(X), model.predict_proba(X)[:, order]


def fit(model, X, y):
    # A convergence failure must not silently become an apparently valid result.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        return model.fit(X, y)


def cv_score(family, params, X, y, folds):
    rows = []
    for fold, (tr, va) in enumerate(folds, 1):
        model = build_model(family, params)
        start = time.perf_counter()
        fit(model, X.iloc[tr], y.iloc[tr])
        elapsed = time.perf_counter() - start
        pred, proba = predict(model, X.iloc[va])
        m, _, _ = scores(y.iloc[va], pred, proba)
        rows.append({'fold': fold, 'train_rows': len(tr), 'cv_rows': len(va),
                     'fit_seconds': elapsed, **m})
    return rows


def run(root, relative=DEFAULT_OUTPUT):
    root = Path(root).resolve()
    # Validate inputs before reserving an output directory. No holdout reads.
    data = {fs: load_split(root, 'train', feature_set=fs) for fs in FEATURE_SETS}
    assert data['pregame'][2].id.equals(data['combined'][2].id)
    X, y, meta = data['combined']
    folds = list(chronological_cv(meta, n_splits=3, gap_days=1))
    out = reserve_output(root, relative)
    source_paths = [root / f'data/plan_a20/splits/{prefix}_{split}.csv'
                    for split in ['train', 'validation'] for prefix in ['X', 'y', 'meta']]
    source_paths += [root / 'artifacts/plan_a20/feature_contract.json',
                     root / 'src/baseline_models20.py', root / 'src/plan_a20_interface.py',
                     root / 'src/baseline_report20.py', root / 'run_baseline_models20.py',
                     root / 'src/model_interface.py', root / 'requirements.txt']
    protocol = {
        'created_utc': now(), 'prediction_ply': 20, 'labels': LABELS,
        'target': 'Original three classes; all draws retained',
        'feature_sets': FEATURE_SETS, 'families': FAMILIES, 'seed': SEED,
        'candidate_grids': {f: specifications(f)[1] for f in FAMILIES},
        'initial_parameters': {f: build_model(f).named_steps['model'].get_params() for f in FAMILIES},
        'initial_policy': 'Repository starter settings, not necessarily sklearn defaults',
        'selection': 'Per family AND feature set: highest mean training CV Macro-F1; '
                     'tie: lower mean CV log loss, then lower candidate ID. Validation cannot change selection.',
        'cv': 'Three expanding chronological training folds; whole UTC days; 24h gap; fresh Pipeline per fit',
        'preprocessing': 'Median imputation and train-fitted VarianceThreshold; StandardScaler for LR and GaussianNB only',
        'nb_priors': 'None=empirical training priors; [1/3]*3=uniform, not class_weight',
        'evaluation': 'Initial and CV-selected models fit on TRAIN ONLY; fixed validation; probability argmax',
        'test_policy': 'No test files or prior test predictions loaded; no finalize stage',
        'limitations': 'Historical test already seen before this extension; validation repeatedly used. '
                       'Grid sizes differ and this is a bounded search, not a global optimum claim.',
        'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in
                     ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'matplotlib', 'chess']},
        'source_sha256': {p.relative_to(root).as_posix(): sha256(p) for p in source_paths},
    }
    save_json(protocol, out / 'artifacts/protocol.json')
    pd.DataFrame([{'fold': i, 'train_rows': len(tr), 'cv_rows': len(va),
                   'train_last_start': meta.iloc[tr].created_at_utc.max(),
                   'cv_first_start': meta.iloc[va].created_at_utc.min()}
                  for i, (tr, va) in enumerate(folds, 1)]).to_csv(out / 'tables/cv_folds.csv', index=False)
    cv_rows, summaries, chosen = [], [], []
    for fs in FEATURE_SETS:
        X, y, _ = data[fs]
        for family in FAMILIES:
            initial, grid = specifications(family)
            cache = {}
            for candidate, params in [(-1, initial), *enumerate(grid)]:
                key = parameter_key(family, params)
                reused = key in cache
                if not reused:
                    cache[key] = cv_score(family, params, X, y, folds)
                values = cache[key]
                for value in values:
                    cv_rows.append({'feature_set': fs, 'family': family, 'candidate': candidate,
                                    'role': 'initial' if candidate == -1 else 'search',
                                    'reused_cv': reused, **value})
                summaries.append({'feature_set': fs, 'family': family, 'candidate': candidate,
                                  'parameters': json.dumps(params, sort_keys=True),
                                  'cv_macro_f1': float(np.mean([v['macro_f1'] for v in values])),
                                  'cv_macro_f1_std': float(np.std([v['macro_f1'] for v in values], ddof=1)),
                                  'cv_accuracy': float(np.mean([v['accuracy'] for v in values])),
                                  'cv_log_loss': float(np.mean([v['log_loss'] for v in values]))})
            if grid:
                table = pd.DataFrame([s for s in summaries if s['feature_set'] == fs
                                      and s['family'] == family and s['candidate'] >= 0])
                best = select_candidate(table[['candidate', 'cv_macro_f1', 'cv_log_loss']])
                chosen.append({'feature_set': fs, 'family': family, 'candidate': best,
                               'parameters': grid[best]})
            pd.DataFrame(cv_rows).to_csv(out / 'tables/cv_fold_metrics.csv', index=False)
            pd.DataFrame(summaries).to_csv(out / 'tables/cv_summary.csv', index=False)
            print(f'CV complete: {fs}/{family} ({len(grid)} search candidates)', flush=True)
    frozen = {'frozen_utc': now(), 'selected': chosen, 'test_loaded': False,
              'validation_labels_loaded_during_selection': False,
              'protocol_sha256': sha256(out / 'artifacts/protocol.json')}
    save_json(frozen, out / 'artifacts/frozen_config.json')
    # Only now read validation labels and fit/export preselected models.
    val = {fs: load_split(root, 'validation', feature_set=fs) for fs in FEATURE_SETS}
    assert val['pregame'][2].id.equals(val['combined'][2].id)
    assert set(meta.id).isdisjoint(val['combined'][2].id)
    results, per_class, group_rows = [], [], []
    for fs in FEATURE_SETS:
        X, y, _ = data[fs]
        Xv, yv, mv = val[fs]
        for family in FAMILIES:
            configs = [('initial', -1, specifications(family)[0])]
            configs += [('tuned', c['candidate'], c['parameters']) for c in chosen
                        if c['feature_set'] == fs and c['family'] == family]
            for stage, candidate, params in configs:
                slug = f'{fs}_{family}_{stage}'
                model = build_model(family, params)
                start = time.perf_counter()
                fit(model, X, y)
                seconds = time.perf_counter() - start
                pred, proba = predict(model, Xv)
                m, report, cm = scores(yv, pred, proba)
                tp, tprob = predict(model, X)
                tm, _, _ = scores(y, tp, tprob)
                cvs = next(s for s in summaries if s['feature_set'] == fs
                           and s['family'] == family and s['candidate'] == candidate)
                results.append({'slug': slug, 'feature_set': fs, 'family': family,
                                'stage': stage, 'candidate': candidate, 'fit_seconds': seconds,
                                'train_accuracy': tm['accuracy'], 'train_macro_f1': tm['macro_f1'],
                                'cv_macro_f1': cvs['cv_macro_f1'], 'cv_accuracy': cvs['cv_accuracy'],
                                'train_validation_macro_f1_gap': tm['macro_f1'] - m['macro_f1'],
                                'parameters': json.dumps(params, sort_keys=True), **m})
                for label in LABELS:
                    per_class.append({'slug': slug, 'class': label, **report[label]})
                for name, mask in [('rating_gap_le100', Xv.abs_rating_diff.le(100)),
                                   ('rating_gap_gt100', Xv.abs_rating_diff.gt(100))]:
                    gm, _, _ = scores(yv.loc[mask], pred[mask], proba[mask])
                    group_rows.append({'slug': slug, 'group': name, 'rows': int(mask.sum()), **gm})
                pd.DataFrame(cm, index=pd.Index(LABELS, name='actual'),
                             columns=pd.Index(LABELS, name='predicted')).to_csv(out / f'tables/{slug}_confusion.csv')
                predictions = mv[['id', 'created_at_utc']].copy()
                predictions['actual'] = yv.to_numpy()
                predictions['predicted'] = pred
                for i, label in enumerate(LABELS):
                    predictions[f'prob_{label}'] = proba[:, i]
                predictions.to_csv(out / f'tables/{slug}_predictions.csv', index=False)
                joblib.dump(model, out / f'models/{slug}.joblib', compress=3)
                print(f'VALIDATION {slug}: Accuracy={m["accuracy"]:.4f}, '
                      f'Macro-F1={m["macro_f1"]:.4f}, Draw recall={m["draw_recall"]:.4f}', flush=True)
    result = pd.DataFrame(results)
    result.to_csv(out / 'tables/validation_metrics.csv', index=False)
    pd.DataFrame(per_class).to_csv(out / 'tables/validation_per_class.csv', index=False)
    pd.DataFrame(group_rows).to_csv(out / 'tables/validation_subgroups.csv', index=False)
    comparison = result[result.family != 'dummy'].pivot(
        index=['feature_set', 'family'], columns='stage', values=['accuracy', 'macro_f1', 'draw_recall', 'log_loss'])
    comparison.columns = ['_'.join(c) for c in comparison.columns]
    for metric in ['accuracy', 'macro_f1', 'draw_recall', 'log_loss']:
        comparison[f'{metric}_delta'] = comparison[f'{metric}_tuned'] - comparison[f'{metric}_initial']
    comparison.reset_index().to_csv(out / 'tables/tuning_changes.csv', index=False)
    for path, expected in protocol['source_sha256'].items():
        if sha256(root / path) != expected:
            raise AssertionError(f'Source changed during run: {path}')
    from .baseline_report20 import draw_figures, write_report, verify_results
    draw_figures(out)
    write_report(out)
    verification = verify_results(root, out, reload_models=True)
    save_json(verification, out / 'artifacts/result_verification.json')
    save_json({'completed_utc': now(), 'test_loaded': False,
               'validation_rows': len(val['combined'][1]), 'train_rows': len(data['combined'][1]),
               'frozen_config_sha256': sha256(out / 'artifacts/frozen_config.json'),
               'validation_metrics_sha256': sha256(out / 'tables/validation_metrics.csv'),
               'verification_checks': verification['check_count']}, out / 'artifacts/development_complete.json')
    inventory = [{'path': p.relative_to(out).as_posix(), 'sha256': sha256(p)}
                 for p in sorted(out.rglob('*')) if p.is_file()
                 and not any(s in p.parts for s in ['matplotlib_cache', '__pycache__'])]
    save_json(inventory, out / 'artifacts/output_hashes.json')
    print(f'Complete: {out}; verified {verification["check_count"]} checks; no test evaluation.', flush=True)


def main(root):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(root, args.output)
