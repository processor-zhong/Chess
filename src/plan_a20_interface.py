"""Model API for Plan A. Original pre-game-only API is left intact."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss
from .model_interface import evaluate, chronological_cv
from .chess_data import LABELS, SEED


def contract(root):
    return json.loads((Path(root)/'artifacts/plan_a20/feature_contract.json').read_text())


def load_split(root, split='train', feature_set='combined', allow_test=False):
    if split not in ['train','validation','test'] or feature_set not in ['pregame','combined']:
        raise ValueError('Invalid split or feature_set')
    if split == 'test' and not allow_test:
        raise ValueError('Freeze choices first; final evaluation requires allow_test=True')
    root = Path(root)
    c = contract(root)
    folder = root/'data/plan_a20'/('holdout' if split == 'test' else 'splits')
    X = pd.read_csv(folder/f'X_{split}.csv')
    meta = pd.read_csv(folder/f'meta_{split}.csv')
    y = pd.read_csv(folder/f'y_{split}.csv')['winner']
    if list(X.columns) != c['combined_features'] or not len(X)==len(y)==len(meta):
        raise ValueError('Column contract or row alignment violated')
    if not meta.id.is_unique or not np.isfinite(X.to_numpy()).all() or not y.isin(LABELS).all():
        raise ValueError('Invalid IDs/features/labels')
    cols = c['pregame_features'] if feature_set == 'pregame' else c['combined_features']
    return X[cols].copy(), y, meta


def make_model(name='logistic', class_weight=None, **params):
    if name == 'logistic':
        defaults = dict(max_iter=3000, C=1.0, class_weight=class_weight, random_state=SEED)
        cls = LogisticRegression
    elif name == 'random_forest':
        defaults = dict(n_estimators=300, min_samples_leaf=5, class_weight=class_weight,
                        random_state=SEED, n_jobs=-1)
        cls = RandomForestClassifier
    elif name == 'gradient_boosting':
        defaults = dict(max_iter=150, max_leaf_nodes=15, early_stopping=False,
                        class_weight=class_weight, random_state=SEED)
        cls = HistGradientBoostingClassifier
    elif name == 'dummy':
        defaults = dict(strategy='most_frequent')
        cls = DummyClassifier
    else:
        raise ValueError('Unknown model')
    defaults.update(params)
    steps = [('imputer', SimpleImputer(strategy='median')), ('nonconstant', VarianceThreshold())]
    if name == 'logistic': steps.append(('scaler', StandardScaler()))
    steps.append(('model', cls(**defaults)))
    return Pipeline(steps)


def evaluate_model(model, X, y):
    metrics, per_class, cm = evaluate(y, model.predict(X))
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        order = [list(model.classes_).index(label) for label in LABELS]
        metrics['log_loss'] = log_loss(y, proba[:,order], labels=LABELS)
    return metrics, per_class, cm
