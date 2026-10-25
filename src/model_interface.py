"""A shared input/evaluation contract for the next group members."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
from .chess_data import FEATURES, LABELS, SEED, build_preprocessor


def load_split(root, split='train', allow_test=False):
    if split not in ['train', 'validation', 'test']:
        raise ValueError('split must be train, validation or test')
    if split == 'test' and not allow_test:
        raise ValueError('Freeze model choices first; then explicitly set allow_test=True.')
    folder = Path(root) / ('data/holdout' if split == 'test' else 'data/splits')
    X = pd.read_csv(folder / f'X_{split}.csv')
    y = pd.read_csv(folder / f'y_{split}.csv')['winner']
    meta = pd.read_csv(folder / f'meta_{split}.csv')
    if list(X.columns) != FEATURES or not len(X) == len(y) == len(meta):
        raise ValueError('Input feature contract or row alignment violated.')
    return X, y, meta


def make_model(name='logistic', class_weight=None):
    # These are starter settings, not selected or tuned hyperparameters.
    if name == 'dummy':
        model = DummyClassifier(strategy='most_frequent')
    elif name == 'logistic':
        model = LogisticRegression(max_iter=2000, C=1.0, class_weight=class_weight,
                                   random_state=SEED)
    elif name == 'random_forest':
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       class_weight=class_weight, random_state=SEED, n_jobs=-1)
    elif name == 'gradient_boosting':
        # Disable random internal validation so the chronological protocol is retained.
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.1,
                    max_leaf_nodes=15, l2_regularization=1.0, early_stopping=False,
                    class_weight=class_weight, random_state=SEED)
    else:
        raise ValueError('Unknown model name')
    return Pipeline([('preprocess', build_preprocessor(scale=name == 'logistic')),
                     ('model', model)])


def evaluate(y_true, y_pred):
    """Fixed class order: black, draw, white, including absent predictions."""
    metrics = {'accuracy': accuracy_score(y_true, y_pred),
               'macro_f1': f1_score(y_true, y_pred, labels=LABELS,
                                    average='macro', zero_division=0),
               'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)}
    per_class = pd.DataFrame(classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)).T
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS),
                      index=pd.Index(LABELS, name='actual'),
                      columns=pd.Index(LABELS, name='predicted'))
    return metrics, per_class, cm


def chronological_cv(meta, n_splits=3, gap_days=1):
    """Expanding folds on TRAIN, approximately balanced by game count.

    Assign whole UTC days to cumulative-count buckets; never split a day.
    Class labels do not define the buckets. Calendar durations can differ.
    Use this iterator with the unfitted Pipeline, never pre-scaled matrices.
    """
    if not isinstance(n_splits, int) or n_splits < 1 or gap_days < 0:
        raise ValueError('n_splits must be a positive integer and gap_days nonnegative.')
    times = pd.to_datetime(meta.created_at_utc, utc=True, format="ISO8601")
    days = times.dt.floor('D')
    daily = days.value_counts().sort_index()
    start_share = (daily.cumsum() - daily) / daily.sum()
    bucket = np.minimum((start_share * (n_splits + 1)).astype(int), n_splits).to_numpy()
    blocks = [daily.index[bucket == k].to_numpy() for k in range(n_splits + 1)]
    if any(len(block) == 0 for block in blocks):
        raise ValueError('Not enough populated count buckets; reduce n_splits.')
    for i in range(1, n_splits + 1):
        valid_start = pd.Timestamp(blocks[i][0])
        completed = pd.to_datetime(meta.last_move_at_utc, utc=True, format="ISO8601").lt(valid_start)
        train_ix = np.flatnonzero(times.lt(valid_start - pd.Timedelta(days=gap_days)) & completed)
        valid_ix = np.flatnonzero(days.isin(blocks[i]))
        if len(train_ix) == 0 or len(valid_ix) == 0:
            raise ValueError('Empty fold after temporal gap.')
        yield train_ix, valid_ix
