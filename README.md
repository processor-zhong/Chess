# Progressive Chess Outcome Prediction

This CA6001 project predicts a chess game's three-class result—Black win, draw, or White win—before play and as moves become available.

The main in-game model jointly uses:

- pre-match ratings, rated status, and time control;
- four Stockfish 19 indicators: `score_cp`, `score_mate`, `wdl_wins`, and `wdl_draws`; and
- three transparent manual features: `ply`, `material_total`, and `material_diff`.

A supervised CNN provides an end-to-end comparison using the board, pre-match features, current ply, and recorded game result. It does not use Stockfish targets or self-play.

## Main results

The cleaned dataset contains 19,113 unique games. The selected pre-match model reaches 52.3% accuracy and 42.7% macro-F1. Across all in-game test snapshots, the joint Stockfish-assisted model reaches 68.7% accuracy and 54.5% macro-F1, compared with 63.9%/48.8% without Stockfish and 61.0%/47.8% for the CNN.

At the last recorded position, pure-prediction performance reaches 89.9% accuracy and 77.9% macro-F1. The known result is never substituted into the curve.

## Repository contents

- `CA6001_chess_outcome_prediction.ipynb` — complete, executed analysis.
- `docs/CA6001_Final_Report_Chess_Outcome_Prediction.docx` — final English report.
- `figures/` — seven 300-DPI figures exported by the notebook.
- `vendor/stockfish/` — Stockfish 19 Linux x86-64 binary, licence, and citation metadata.
- `pyproject.toml`, `uv.lock`, and `.python-version` — reproducible Python 3.11 environment.

The notebook downloads the public [Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess) when `data/games.csv` is absent. On Linux x86-64 it can also download the pinned Stockfish 19 binary when the vendored executable is absent. Other platforms should set `STOCKFISH_EXECUTABLE` to a compatible Stockfish binary.

Stockfish results are cached in `data/stockfish_features.csv`. The cache key includes the position, game and ply, engine binary hash, and node budget; rerunning does not append duplicates.

## Run with uv

From the repository root:

```bash
UV_CACHE_DIR=.uv-cache uv sync --locked
UV_CACHE_DIR=.uv-cache uv run jupyter lab
```

Open `CA6001_chess_outcome_prediction.ipynb` and select **Run All**.

For non-interactive execution:

```bash
UV_CACHE_DIR=.uv-cache uv run jupyter execute \
  --inplace --timeout=-1 CA6001_chess_outcome_prediction.ipynb
```

The first execution takes longer because it analyses and caches approximately 146,000 positions with Stockfish. Later executions reuse the cache.

## Run with standard pip

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  ipykernel jupyterlab matplotlib nbclient nbformat numpy pandas \
  python-chess scikit-learn seaborn python-docx
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and set the engine path before running:

```powershell
$env:STOCKFISH_EXECUTABLE = "C:\path\to\stockfish.exe"
```

## Reproducibility

- Random seed: `6001`.
- Game split: stratified 70% training, 15% validation, and 15% test.
- Stockfish configuration: version 19, one thread, 1,000 nodes, WDL enabled.
- Dataset SHA-256: `e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129`.
- Stockfish SHA-256: `0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19`.
- Percentage progress is used only for retrospective test reporting and is never a model input.
