"""Adapted from processor-zhong/Chess feature@5f86a39 notebook cells6/15.

The progress argument/feature and unused neural board-plane encoder are removed.
Percentages below schedule retrospective evaluation only, never model inputs.
"""
import numpy as np
import pandas as pd
import chess

VALID_WINNERS = {'black', 'draw', 'white'}
LABEL_TO_INT = {'black': 0, 'draw': 1, 'white': 2}
CLASS_NAMES = np.array(['black', 'draw', 'white'])

def clean_games(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply explicit quality rules and return clean rows plus an audit trail."""
    audit = {}
    work = frame.copy()
    audit['raw_rows'] = len(work)
    work = work.drop_duplicates(subset='id', keep='first')
    audit['duplicate_ids_removed'] = audit['raw_rows'] - len(work)
    valid_target = work['winner'].isin(VALID_WINNERS)
    valid_ratings = work['white_rating'].between(100, 4000) & work['black_rating'].between(100, 4000)
    valid_moves = work['moves'].fillna('').str.strip().ne('')
    parsed_time = work['increment_code'].astype(str).str.extract('^(\\d+)\\+(\\d+)$').astype(float)
    valid_time = parsed_time.notna().all(axis=1)
    keep = valid_target & valid_ratings & valid_moves & valid_time
    audit['invalid_target_removed'] = int((~valid_target).sum())
    audit['invalid_rating_removed'] = int((~valid_ratings).sum())
    audit['empty_move_list_removed'] = int((~valid_moves).sum())
    audit['invalid_time_control_removed'] = int((~valid_time).sum())
    work = work.loc[keep].copy().reset_index(drop=True)
    parsed_time = parsed_time.loc[keep].reset_index(drop=True)
    work['base_minutes'] = parsed_time[0].astype(int)
    work['increment_seconds'] = parsed_time[1].astype(int)
    work['rated_int'] = work['rated'].astype(int)
    work['move_count'] = work['moves'].str.split().str.len()
    work['rating_diff'] = work['white_rating'] - work['black_rating']
    work['rating_mean'] = (work['white_rating'] + work['black_rating']) / 2
    work['rating_gap'] = work['rating_diff'].abs()
    work['log_base_minutes'] = np.log1p(work['base_minutes'])
    work['white_elo_expectation'] = 1 / (1 + 10 ** (-work['rating_diff'] / 400))
    work['target'] = work['winner'].map(LABEL_TO_INT).astype(int)
    audit['clean_rows'] = len(work)
    return (work, pd.Series(audit, name='count'))
PROGRESS_POINTS = (0.1, 0.25, 0.5, 0.75, 1.0)
PIECE_VALUES = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25, chess.ROOK: 5.0, chess.QUEEN: 9.0}
PIECE_NAMES = {chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop', chess.ROOK: 'rook', chess.QUEEN: 'queen'}
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

def legal_move_count(board: chess.Board, color: chess.Color) -> int:
    """Count legal moves after setting the side to move on a copied board."""
    copy = board.copy(stack=False)
    copy.turn = color
    return copy.legal_moves.count()

def pawn_structure(board: chess.Board, color: chess.Color) -> tuple[int, int, int]:
    """Return doubled, isolated, and passed pawn counts for one color."""
    pawn_squares = list(board.pieces(chess.PAWN, color))
    files = [chess.square_file(square) for square in pawn_squares]
    doubled = sum((max(files.count(file_index) - 1, 0) for file_index in set(files)))
    isolated = sum((not any((abs(chess.square_file(other) - chess.square_file(square)) == 1 for other in pawn_squares)) for square in pawn_squares))
    enemy_pawns = list(board.pieces(chess.PAWN, not color))
    passed = 0
    for square in pawn_squares:
        file_index, rank_index = (chess.square_file(square), chess.square_rank(square))
        blockers = []
        for enemy in enemy_pawns:
            enemy_file, enemy_rank = (chess.square_file(enemy), chess.square_rank(enemy))
            ahead = enemy_rank > rank_index if color == chess.WHITE else enemy_rank < rank_index
            blockers.append(ahead and abs(enemy_file - file_index) <= 1)
        passed += not any(blockers)
    return (int(doubled), int(isolated), int(passed))

def king_shield(board: chess.Board, color: chess.Color) -> int:
    """Count friendly pawns on adjacent squares around the king."""
    king_square = board.king(color)
    if king_square is None:
        return 0
    return sum((board.piece_at(square) == chess.Piece(chess.PAWN, color) for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square])))

def board_features(board: chess.Board) -> dict[str, float]:
    """Create interpretable material and positional features from a board."""
    features: dict[str, float] = {}
    material_balance = 0.0
    for piece_type, piece_name in PIECE_NAMES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        difference = white_count - black_count
        features[f'{piece_name}_diff'] = difference
        material_balance += PIECE_VALUES[piece_type] * difference
    features['material_balance'] = material_balance
    features['bishop_pair_diff'] = int(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2) - int(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2)
    features['mobility_diff'] = legal_move_count(board, chess.WHITE) - legal_move_count(board, chess.BLACK)
    features['center_control_diff'] = sum((len(board.attackers(chess.WHITE, square)) - len(board.attackers(chess.BLACK, square)) for square in CENTER_SQUARES))
    features['king_shield_diff'] = king_shield(board, chess.WHITE) - king_shield(board, chess.BLACK)
    white_structure = pawn_structure(board, chess.WHITE)
    black_structure = pawn_structure(board, chess.BLACK)
    for name, white_value, black_value in zip(('doubled_pawn_diff', 'isolated_pawn_diff', 'passed_pawn_diff'), white_structure, black_structure):
        features[name] = white_value - black_value
    features['castling_rights_diff'] = int(board.has_kingside_castling_rights(chess.WHITE)) + int(board.has_queenside_castling_rights(chess.WHITE)) - int(board.has_kingside_castling_rights(chess.BLACK)) - int(board.has_queenside_castling_rights(chess.BLACK))
    features['white_in_check'] = int(board.turn == chess.WHITE and board.is_check())
    features['black_in_check'] = int(board.turn == chess.BLACK and board.is_check())
    features['white_to_move'] = int(board.turn == chess.WHITE)
    return features
