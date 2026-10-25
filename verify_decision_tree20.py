"""Check saved decision-tree development results without training or test access."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from src.chess_data import sha256
from src.baseline_models20 import select_candidate
from src.plan_a20_interface import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='modelling_baselines/run_20261025_tree_01')
    parser.add_argument('--reload-models', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root / args.output
    count = 0

    def check(ok, message):
        nonlocal count
        if not bool(ok):
            raise AssertionError(message)
        count += 1

    for item in json.loads((out / 'artifacts/output_hashes.json').read_text(encoding='utf-8')):
        check(sha256(out / item['path']) == item['sha256'], item['path'])
    frozen = json.loads((out / 'artifacts/frozen_config.json').read_text(encoding='utf-8'))
    protocol = json.loads((out / 'artifacts/protocol.json').read_text(encoding='utf-8'))
    check(not frozen['test_loaded'] and not frozen['validation_labels_loaded'], 'Selection boundary')
    check(sha256(out / 'artifacts/protocol.json') == frozen['protocol_sha256'], 'Protocol hash')
    for path, expected in protocol['source_sha256'].items():
        check(sha256(root / path) == expected, path)
    cv = pd.read_csv(out / 'tables/cv_summary.csv')
    for choice in frozen['selected']:
        block = cv[(cv.feature_set == choice['feature_set']) & (cv.candidate >= 0)]
        check(select_candidate(block[['candidate', 'cv_macro_f1', 'cv_log_loss']]) == choice['candidate'],
              'Training CV selection: ' + choice['feature_set'])
    metrics = pd.read_csv(out / 'tables/validation_metrics.csv')
    check(len(metrics) == 4 and not metrics.duplicated(['feature_set', 'stage']).any(), 'Four evaluations')
    labels = ['black', 'draw', 'white']
    for r in metrics.itertuples():
        slug = f'{r.feature_set}_{r.stage}'
        X, y, meta = load_split(root, 'validation', feature_set=r.feature_set)
        p = pd.read_csv(out / f'tables/{slug}_predictions.csv')
        prob = p[[f'prob_{c}' for c in labels]].to_numpy()
        check(p.id.equals(meta.id) and p.actual.equals(y), 'Alignment: ' + slug)
        check(np.isfinite(prob).all() and (prob >= 0).all() and np.allclose(prob.sum(axis=1), 1), 'Probabilities: ' + slug)
        check(np.array_equal(np.asarray(labels)[prob.argmax(axis=1)], p.predicted), 'Argmax: ' + slug)
        expected = dict(accuracy=accuracy_score(y, p.predicted),
                        macro_f1=f1_score(y, p.predicted, labels=labels, average='macro', zero_division=0),
                        balanced_accuracy=balanced_accuracy_score(y, p.predicted),
                        log_loss=log_loss(y, prob, labels=labels),
                        macro_auc=roc_auc_score(y, prob, labels=labels, multi_class='ovr', average='macro'))
        for name, value in expected.items():
            check(np.isclose(value, getattr(r, name), rtol=1e-10, atol=1e-10), name + ': ' + slug)
        cm = confusion_matrix(y, p.predicted, labels=labels)
        check(np.array_equal(cm, pd.read_csv(out / f'tables/{slug}_confusion.csv', index_col=0)), 'Confusion: ' + slug)
        check(np.isclose(cm[1, 1] / cm[1].sum(), r.draw_recall), 'Draw recall: ' + slug)
        if args.reload_models:
            model = joblib.load(out / f'models/{slug}.joblib')
            check(np.allclose(model.predict_proba(X), prob, rtol=1e-10, atol=1e-10), 'Reloaded probabilities: ' + slug)
            tree = model.named_steps['model']
            check(tree.get_depth() == r.actual_depth and tree.get_n_leaves() == r.leaf_count, 'Tree size: ' + slug)
    print(f'Passed {count} decision-tree checks; no training or test access.')


if __name__ == '__main__':
    main()
