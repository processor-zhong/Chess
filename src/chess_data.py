"""Prepare the fixed Kaggle snapshot without learning from the holdout set."""
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAW_COLUMNS = ['id', 'rated', 'created_at', 'last_move_at', 'turns',
               'victory_status', 'winner', 'increment_code', 'white_id',
               'white_rating', 'black_id', 'black_rating', 'moves',
               'opening_eco', 'opening_name', 'opening_ply']
NUMERIC_FEATURES = ['rating_diff', 'avg_rating', 'abs_rating_diff',
                    'base_time_min', 'increment_seconds']
FEATURES = NUMERIC_FEATURES + ['rated']
LABELS = ['black', 'draw', 'white']
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}
RAW_SHA256 = 'e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129'
SEED = 42


def save_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=str), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raw(root):
    path = Path(root) / 'data/raw/games.csv'
    if sha256(path) != RAW_SHA256:
        raise ValueError('Raw snapshot differs from the verified Kaggle file.')
    tokens = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw = pd.read_csv(path)
    if list(raw.columns) != RAW_COLUMNS or raw.shape != (20058, 16):
        raise ValueError('Unexpected input schema or row count.')
    return raw, tokens


def audit_raw(raw, tokens):
    profile = pd.DataFrame({
        'column': raw.columns,
        'dtype': [str(raw[c].dtype) for c in raw],
        'missing': [int(raw[c].isna().sum()) for c in raw],
        'blank': [int(tokens[c].str.strip().eq('').sum()) for c in raw],
        'unique_values': [int(raw[c].nunique()) for c in raw],
    })
    after_exact = raw.drop_duplicates()
    repeated = after_exact[after_exact.id.duplicated(keep=False)]
    varying = repeated.groupby('id').nunique().gt(1).sum().to_dict()
    audit = {
        'raw_rows': len(raw), 'raw_columns': len(raw.columns),
        'missing_cells': int(raw.isna().sum().sum()),
        'blank_cells': int(profile.blank.sum()),
        'exact_duplicate_surplus': int(raw.duplicated().sum()),
        'duplicate_id_surplus': int(raw.id.duplicated().sum()),
        'repeated_ids': int(raw.loc[raw.id.duplicated(False), 'id'].nunique()),
        'ids_with_nonidentical_records_after_exact_dedup': int(repeated.id.nunique()),
        'columns_varying_within_repeated_ids': {k: int(v) for k, v in varying.items()},
        'same_player_rows': int(raw.white_id.eq(raw.black_id).sum()),
        'zero_recorded_duration_rows': int(raw.created_at.eq(raw.last_move_at).sum()),
        'negative_recorded_duration_rows': int(raw.last_move_at.lt(raw.created_at).sum()),
        'moves_count_mismatches': int(raw.moves.str.split().str.len().ne(raw.turns).sum()),
        'opening_longer_than_game': int(raw.opening_ply.gt(raw.turns).sum()),
        'outoftime_draw_rows_retained': int((raw.victory_status.eq('outoftime') & raw.winner.eq('draw')).sum()),
    }
    return audit, profile


def clean_games(raw, tokens):
    """One game per ID; prefer timestamp tokens with more significant digits.

    Significant digits are a resolution proxy, not a guarantee of truth.
    Original CSV line number breaks ties. No outcome is used for ranking.
    """
    data = raw.copy()
    data['source_csv_row'] = np.arange(len(data)) + 2
    for name in ['created_at', 'last_move_at']:
        data[f'{name}_sig_digits'] = tokens[name].map(
            lambda x: len(Decimal(x).normalize().as_tuple().digits))

    non_time = [c for c in RAW_COLUMNS if c not in ['id', 'created_at', 'last_move_at']]
    conflict_ids = set(data.groupby('id')[non_time].nunique(dropna=False)
                       .gt(1).any(axis=1).loc[lambda x: x].index)
    ranked = data.sort_values(
        ['id', 'created_at_sig_digits', 'last_move_at_sig_digits', 'source_csv_row'],
        ascending=[True, False, False, True], kind='stable')
    canonical = ranked.drop_duplicates('id', keep='first').copy()
    chosen_rows = set(canonical.source_csv_row)
    decisions = data[['source_csv_row', 'id']].copy()
    decisions['decision'] = np.where(decisions.source_csv_row.isin(chosen_rows),
                                     'retained', 'duplicate_id_removed')
    decisions.loc[decisions.id.isin(conflict_ids), 'decision'] = 'conflicting_non_time_fields'

    required_numeric = ['created_at', 'last_move_at', 'white_rating', 'black_rating']
    invalid = (canonical[required_numeric].isna().any(axis=1)
               | ~canonical.winner.isin(LABELS)
               | ~canonical.increment_code.str.fullmatch(r'\d+\+\d+')
               | canonical[['white_rating', 'black_rating']].le(0).any(axis=1)
               | canonical.id.str.strip().eq(''))
    bad_ids = set(canonical.loc[invalid, 'id'])
    same_ids = set(canonical.loc[canonical.white_id.eq(canonical.black_id), 'id'])
    decisions.loc[decisions.id.isin(same_ids) & decisions.decision.eq('retained'),
                  'decision'] = 'same_player_removed'
    decisions.loc[decisions.id.isin(bad_ids) & decisions.decision.eq('retained'),
                  'decision'] = 'invalid_required_value'
    clean = canonical[~canonical.id.isin(conflict_ids | bad_ids | same_ids)].copy()
    clean['created_at'] = clean.created_at.astype('int64')
    clean['last_move_at'] = clean.last_move_at.astype('int64')
    clean['created_at_utc'] = pd.to_datetime(clean.created_at, unit='ms', utc=True)
    clean['last_move_at_utc'] = pd.to_datetime(clean.last_move_at, unit='ms', utc=True)
    clean['created_day_utc'] = clean.created_at_utc.dt.floor('D')
    clean['rated'] = clean.rated.astype('int64')
    control = clean.increment_code.str.split('+', expand=True).astype('int64')
    clean['base_time_min'] = control[0]
    clean['increment_seconds'] = control[1]
    clean['rating_diff'] = clean.white_rating - clean.black_rating
    clean['avg_rating'] = (clean.white_rating + clean.black_rating) / 2
    clean['abs_rating_diff'] = clean.rating_diff.abs()
    clean['y_encoded'] = clean.winner.map(LABEL_MAP)
    clean = clean.sort_values(['created_at', 'id'], kind='stable').reset_index(drop=True)

    groups = data.groupby('id', sort=True)
    duplicate_summary = groups.agg(
        records=('id', 'size'), start_min=('created_at', 'min'),
        start_max=('created_at', 'max'))
    duplicate_summary = duplicate_summary[duplicate_summary.records.gt(1)].copy()
    duplicate_summary['start_disagreement_minutes'] = (
        duplicate_summary.start_max - duplicate_summary.start_min) / 60000
    duplicate_summary['selected_source_csv_row'] = canonical.set_index('id').source_csv_row
    duplicate_summary['non_time_conflict'] = duplicate_summary.index.isin(conflict_ids)
    return clean, decisions, duplicate_summary.reset_index()


def temporal_partition(clean):
    """Approximate 70/15/15 by whole UTC days, then apply a 24 h boundary gap.

    Also exclude earlier-block games whose recorded outcome was not available
    before the next block. These exclusions never use the class label.
    """
    counts = clean.groupby('created_day_utc').size()
    n_before = counts.cumsum() - counts
    val_start = (n_before - len(clean) * .70).abs().idxmin()
    test_start = (n_before - len(clean) * .85).abs().idxmin()
    if val_start >= test_start:
        raise ValueError('Temporal split boundaries collapsed.')
    split = clean[['id', 'source_csv_row', 'created_at', 'created_at_utc',
                   'last_move_at', 'last_move_at_utc', 'created_day_utc',
                   'white_id', 'black_id', 'created_at_sig_digits']].copy()
    split['nominal_split'] = np.select(
        [clean.created_day_utc.lt(val_start), clean.created_day_utc.lt(test_start)],
        ['train', 'validation'], default='test')
    split['split'] = split.nominal_split
    split['exclusion_reason'] = ''
    for earlier, next_start in [('train', val_start), ('validation', test_start)]:
        earlier_mask = split.nominal_split.eq(earlier)
        in_gap = earlier_mask & split.created_at_utc.ge(next_start - pd.Timedelta(days=1))
        split.loc[in_gap, 'split'] = 'excluded'
        split.loc[in_gap, 'exclusion_reason'] = '24h_boundary_gap'
        late = earlier_mask & split.last_move_at_utc.ge(next_start)
        split.loc[late & ~in_gap, 'exclusion_reason'] = 'outcome_not_available_before_next_block'
        split.loc[late, 'split'] = 'excluded'
    info = {
        'method': 'whole UTC days, target 70/15/15, 24h gap before each later block',
        'validation_start_utc': val_start.isoformat(),
        'test_start_utc': test_start.isoformat(), 'gap_hours': 24,
        'boundary_selection_uses': 'created_at only; never outcomes',
        'clean_rows_before_gap': len(clean),
        'nominal_counts': {k: int(v) for k, v in split.nominal_split.value_counts().items()},
        'final_counts': {k: int(v) for k, v in split.split.value_counts().items()},
        'exclusion_counts': split.loc[split.split.eq('excluded'), 'exclusion_reason'].value_counts().to_dict(),
    }
    return split, info


def build_preprocessor(scale=True):
    """Return an UNFITTED transformer. Fit inside each CV fold's Pipeline."""
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    return ColumnTransformer(
        [('numeric', Pipeline(steps), NUMERIC_FEATURES),
         ('binary', SimpleImputer(strategy='most_frequent'), ['rated'])],
        remainder='drop', verbose_feature_names_out=False)


def player_overlap(clean, manifest):
    rows = []
    indexed = clean.set_index('id')
    sets = {}
    for name in ['train', 'validation', 'test']:
        frame = indexed.loc[manifest.loc[manifest.split.eq(name), 'id']]
        sets[name] = set(frame.white_id) | set(frame.black_id)
    for later in ['validation', 'test']:
        frame = indexed.loc[manifest.loc[manifest.split.eq(later), 'id']]
        white_seen = frame.white_id.isin(sets['train'])
        black_seen = frame.black_id.isin(sets['train'])
        rows.append({'split': later, 'games': len(frame),
                     'unique_players': len(sets[later]),
                     'players_also_in_train': len(sets[later] & sets['train']),
                     'both_players_seen_in_train_games': int((white_seen & black_seen).sum()),
                     'at_least_one_seen_in_train_games': int((white_seen | black_seen).sum()),
                     'neither_seen_in_train_games': int((~white_seen & ~black_seen).sum()),
                     'at_least_one_seen_in_train_pct': float((white_seen | black_seen).mean() * 100)})
    return pd.DataFrame(rows)


def export_data(root, raw, tokens, audit, profile, clean, decisions, dup, manifest, split_info):
    root = Path(root)
    for folder in ['data/processed', 'data/splits', 'data/holdout', 'artifacts', 'tables', 'figures']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    profile.to_csv(root / 'tables/raw_column_audit.csv', index=False)
    decisions.to_csv(root / 'tables/row_decisions.csv', index=False)
    dup.to_csv(root / 'tables/duplicate_id_audit.csv', index=False)
    clean.to_csv(root / 'data/processed/clean_games_for_audit.csv', index=False)
    manifest.to_csv(root / 'data/splits/split_manifest.csv', index=False)
    # This clean table contains post-game fields. Use X files for modelling.
    indexed = clean.set_index('id', drop=False)
    pieces = {}
    sizes = []
    for split in ['train', 'validation', 'test']:
        part = indexed.loc[manifest.loc[manifest.split.eq(split), 'id']].reset_index(drop=True)
        pieces[split] = part
        out = root / ('data/holdout' if split == 'test' else 'data/splits')
        part[FEATURES].to_csv(out / f'X_{split}.csv', index=False)
        # Keep labels separate from both inputs and metadata.
        part[['winner']].to_csv(out / f'y_{split}.csv', index=False)
        part[['id', 'source_csv_row', 'created_at_utc', 'created_day_utc', 'last_move_at_utc',
              'white_id', 'black_id']].to_csv(out / f'meta_{split}.csv', index=False)
        sizes.append({'split': split, 'rows': len(part),
                      'share_of_active_pct': len(part) / manifest.split.ne('excluded').sum() * 100,
                      'first_start_utc': part.created_at_utc.min(),
                      'last_start_utc': part.created_at_utc.max(),
                      'last_recorded_outcome_utc': part.last_move_at_utc.max()})
    pd.DataFrame(sizes).to_csv(root / 'tables/split_summary.csv', index=False)
    player_overlap(clean, manifest).to_csv(root / 'tables/player_overlap_audit.csv', index=False)
    for name, scale in [('scaled', True), ('unscaled', False)]:
        processor = build_preprocessor(scale=scale)
        processor.fit(pieces['train'][FEATURES])
        joblib.dump(processor, root / f'artifacts/{name}_preprocessor_train_only.joblib')
        # No fitted test matrix is exported or used in development.
        for split in ['train', 'validation']:
            transformed = pd.DataFrame(processor.transform(pieces[split][FEATURES]), columns=FEATURES)
            transformed.to_csv(root / f'data/splits/X_{split}_{name}.csv', index=False)
    audit.update({
        'unique_ids_before_same_player_filter': int(raw.id.nunique()),
        'clean_rows': len(clean),
        'removed_rows': len(raw) - len(clean),
        'clean_zero_duration_rows_retained': int(clean.created_at.eq(clean.last_move_at).sum()),
        'clean_non_time_conflict_ids': int(dup.non_time_conflict.sum()),
        'max_duplicate_start_disagreement_minutes': float(dup.start_disagreement_minutes.max()),
        'clean_start_min_utc': clean.created_at_utc.min().isoformat(),
        'clean_start_max_utc': clean.created_at_utc.max().isoformat(),
    })
    save_json(audit, root / 'artifacts/data_audit.json')
    save_json(split_info, root / 'artifacts/split_protocol.json')
    save_json({'label_order': LABELS, 'label_to_int': LABEL_MAP,
               'positive_rating_diff_means': 'white has the higher rating',
               'features_in_order': FEATURES, 'units': {'base_time_min': 'minutes',
                   'increment_seconds': 'seconds per move'},
               'test_folder': 'data/holdout',
               'default_loader_exposes_test': False}, root / 'artifacts/feature_contract.json')
    save_json({'python': platform.python_version(),
               'packages': {p: importlib.metadata.version(p) for p in
                            ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'scipy', 'joblib']}},
              root / 'artifacts/environment.json')
    return pieces


def prepare(root):
    raw, tokens = read_raw(root)
    audit, profile = audit_raw(raw, tokens)
    clean, decisions, dup = clean_games(raw, tokens)
    manifest, protocol = temporal_partition(clean)
    pieces = export_data(root, raw, tokens, audit, profile, clean, decisions,
                         dup, manifest, protocol)
    return audit, clean, manifest, pieces
