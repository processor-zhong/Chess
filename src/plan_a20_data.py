"""Plan A: deterministic features from exactly the first 20 legal plies.

The existing clean snapshot and frozen split membership are inputs, not outputs.
No chess engine, future SAN, outcome or opening metadata is used in extraction.
"""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata
import json
import platform

import chess
import numpy as np
import pandas as pd
from .chess_data import FEATURES as PREGAME_FEATURES, LABELS, sha256
from .plan_a20_feature_dictionary import write_feature_dictionary

PLY = 20
PIECES = {'pawn': chess.PAWN, 'knight': chess.KNIGHT, 'bishop': chess.BISHOP,
          'rook': chess.ROOK, 'queen': chess.QUEEN}
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
          chess.ROOK: 5, chess.QUEEN: 9}
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]

def dump_json(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def opening_features(prefix):
    """Accept ONLY a fixed prefix. Deliberately cannot access full rows or labels."""
    if len(prefix) != PLY:
        raise ValueError('Exactly 20 SAN tokens required')
    board = chess.Board()
    castles = {chess.WHITE: 0, chess.BLACK: 0}
    captures = {chess.WHITE: 0, chess.BLACK: 0}
    checks = {chess.WHITE: 0, chess.BLACK: 0}
    for ply, token in enumerate(prefix, 1):
        color = board.turn
        try:
            move = board.parse_san(token)
        except ValueError as exc:
            raise ValueError(f'Invalid SAN at ply {ply}: {token}: {exc}') from exc
        if not move or move not in board.legal_moves:
            raise ValueError(f'Illegal/null move at ply {ply}: {token}')
        castles[color] |= int(board.is_castling(move))
        captures[color] += int(board.is_capture(move))
        board.push(move)
        checks[color] += int(board.is_check())
    if not board.is_valid():
        raise ValueError('Invalid reconstructed position')
    if board.is_game_over(claim_draw=False):
        raise ValueError('Terminal board at cutoff despite subsequent recorded moves')
    features = {}
    for name, color in [('white', chess.WHITE), ('black', chess.BLACK)]:
        pieces = board.occupied_co[color]
        for piece_name, pt in PIECES.items():
            features[f'{name}_{piece_name}_count'] = len(board.pieces(pt, color))
        features[f'{name}_piece_count'] = chess.popcount(pieces)  # includes king
        features[f'{name}_material'] = sum(v * len(board.pieces(pt, color)) for pt, v in VALUES.items())
        home = 0 if color == chess.WHITE else 7
        minors = board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color)
        features[f'{name}_minor_off_home_rank'] = sum(chess.square_rank(s) != home for s in minors)
        features[f'{name}_center_occupancy'] = sum(bool(pieces & chess.BB_SQUARES[s]) for s in CENTER)
        # Includes defended own pieces: attacked center squares, not legal move count.
        features[f'{name}_center_attacked_squares'] = sum(board.is_attacked_by(color, s) for s in CENTER)
        features[f'{name}_attacked_squares'] = sum(board.is_attacked_by(color, s) for s in chess.SQUARES)
        features[f'{name}_pawn_advancement'] = sum(
            max(0, chess.square_rank(s) - 1 if color else 6 - chess.square_rank(s))
            for s in board.pieces(chess.PAWN, color))
        features[f'{name}_castled'] = castles[color]
        features[f'{name}_kingside_right'] = int(board.has_kingside_castling_rights(color))
        features[f'{name}_queenside_right'] = int(board.has_queenside_castling_rights(color))
        king = board.king(color)
        features[f'{name}_king_file'] = chess.square_file(king)
        features[f'{name}_king_rank'] = chess.square_rank(king)
        features[f'{name}_captures_first20'] = captures[color]
        features[f'{name}_checks_given_first20'] = checks[color]
    features['material_diff'] = features['white_material'] - features['black_material']
    features['side_to_move_in_check'] = int(board.is_check())
    return features, board.fen()


def build_plan_a20(root):
    root = Path(root)
    clean_path = root / 'data/processed/clean_games_for_audit.csv'
    manifest_path = root / 'data/splits/split_manifest.csv'
    clean_hash, manifest_hash = sha256(clean_path), sha256(manifest_path)
    clean = pd.read_csv(clean_path)
    manifest = pd.read_csv(manifest_path)
    if not clean.id.is_unique or not manifest.id.is_unique or set(clean.id) != set(manifest.id):
        raise ValueError('Input ID uniqueness/coverage violated')
    indexed = clean.set_index('id', drop=False)
    rows, decisions, meta_rows = [], [], []
    for m in manifest.to_dict('records'):
        row = indexed.loc[m['id']]
        audit = {'id': row.id, 'original_split': m['split'], 'decision': '', 'error': ''}
        if m['split'] == 'excluded':
            audit['decision'] = 'original_boundary_gap'
            decisions.append(audit)
            continue
        # Full length is used only to define the ongoing-game cohort, never as a feature.
        tokens = str(row.moves).split()
        if len(tokens) <= PLY:
            audit['decision'] = 'finished_by_ply20'
            decisions.append(audit)
            continue
        try:
            feat, fen = opening_features(tokens[:PLY])
        except ValueError as exc:
            audit['decision'] = 'invalid_or_terminal_prefix'
            audit['error'] = str(exc)
            decisions.append(audit)
            continue
        audit['decision'] = 'retained'
        decisions.append(audit)
        base = {name: float(row[name]) for name in PREGAME_FEATURES}
        rows.append({'id': row.id, 'split': m['split'], 'winner': row.winner, **base, **feat})
        meta_rows.append({
            'id': row.id, 'source_csv_row': int(row.source_csv_row),
            'created_at_utc': row.created_at_utc, 'last_move_at_utc': row.last_move_at_utc,
            'created_day_utc': row.created_day_utc,
            'white_id': row.white_id, 'black_id': row.black_id,
            'position_fen_at_ply20': fen})
    frame = pd.DataFrame(rows)
    metadata = pd.DataFrame(meta_rows).set_index('id')
    opening = list(opening_features('e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7'.split())[0])
    features = list(PREGAME_FEATURES) + opening
    paths = {'base': root/'data/plan_a20', 'tables': root/'tables/plan_a20',
             'artifacts': root/'artifacts/plan_a20'}
    for path in paths.values(): path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decisions).to_csv(paths['tables']/'row_decisions.csv', index=False)
    pd.DataFrame(decisions).groupby(['original_split', 'decision']).size().rename('rows').reset_index().to_csv(
        paths['tables']/'processing_summary.csv', index=False)
    summary = []
    for split in ['train', 'validation', 'test']:
        part = frame[frame.split.eq(split)].reset_index(drop=True)
        folder = paths['base']/('holdout' if split == 'test' else 'splits')
        folder.mkdir(parents=True, exist_ok=True)
        part[features].to_csv(folder/f'X_{split}.csv', index=False)
        part[list(PREGAME_FEATURES)].to_csv(folder/f'X_{split}_pregame.csv', index=False)
        part[['winner']].to_csv(folder/f'y_{split}.csv', index=False)
        metadata.loc[part.id].reset_index().to_csv(folder/f'meta_{split}.csv', index=False)
        summary.append({'split': split, 'original_rows': int(manifest.split.eq(split).sum()),
                        'retained_rows': len(part),
                        'first_start_utc': metadata.loc[part.id, 'created_at_utc'].iloc[0],
                        'last_start_utc': metadata.loc[part.id, 'created_at_utc'].iloc[-1]})
    pd.DataFrame(summary).to_csv(paths['tables']/'split_summary.csv', index=False)
    # Only training labels are summarized. Test y is exported, never scored or explored.
    train = frame[frame.split.eq('train')]
    train.winner.value_counts().reindex(LABELS, fill_value=0).rename_axis('winner').rename('rows').reset_index().to_csv(
        paths['tables']/'train_class_balance.csv', index=False)
    train[features].describe().T.to_csv(paths['tables']/'train_feature_summary.csv')
    constants = [f for f in features if train[f].nunique() <= 1]
    dump_json({'schema_version': 1, 'prediction_ply': PLY, 'label_order': LABELS,
               'label_to_int': {v:i for i,v in enumerate(LABELS)},
               'pregame_features': list(PREGAME_FEATURES), 'opening_features': opening,
               'combined_features': features, 'constant_features_on_train': constants,
               'constant_handling': 'Retained in export; removed by train-fitted VarianceThreshold in Pipeline',
               'source_clean_sha256': clean_hash, 'source_manifest_sha256': manifest_hash,
               'cohort': 'Existing non-gap split; >20 SAN tokens; first 20 legal; board not terminal at cutoff',
               'holdout_policy': 'Structure/feature extraction only; no model scoring or label EDA',
               'side_to_move': 'Always white at ply 20, omitted as a constant',
               'forbidden_inputs': ['winner','turns','last_move_at','victory_status','moves',
                                    'opening_eco','opening_name','opening_ply','id','white_id','black_id',
                                    'position_fen_at_ply20']}, paths['artifacts']/'feature_contract.json')
    write_feature_dictionary(root, json.loads((paths['artifacts']/'feature_contract.json').read_text()))
    decision_counts = Counter(d['decision'] for d in decisions)
    result = {'status': 'completed', 'cutoff_ply': PLY,
              'source_clean_rows': len(clean), 'decisions': dict(decision_counts),
              'splits': summary, 'pregame_feature_count': len(PREGAME_FEATURES),
              'opening_feature_count': len(opening), 'combined_feature_count': len(features),
              'python': platform.python_version(),
              'packages': {p:importlib.metadata.version(p) for p in ['chess','pandas','numpy','scikit-learn']},
              'test_performance_evaluated': False}
    dump_json(result, paths['artifacts']/'processing_audit.json')
    inventory = []
    for folder in [paths['base'], paths['tables']]:
        for f in sorted(folder.rglob('*.csv')):
            inventory.append({'path': str(f.relative_to(root)), 'sha256': sha256(f)})
    dump_json(inventory, paths['artifacts']/'output_hashes.json')
    if sha256(clean_path) != clean_hash or sha256(manifest_path) != manifest_hash:
        raise AssertionError('Original input changed')
    return result
