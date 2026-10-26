# Human-Context Chess Outcome Prediction

This CA6001 project studies whether human context improves chess-outcome prediction beyond an objective engine evaluation. Stockfish can estimate the strength of a position, but its WDL probabilities describe engine play. Human results may also depend on player strength, time control, and the practical difficulty of converting an advantage.

The controlled experiment compares three in-game predictors:

- **Direct Stockfish WDL:** no fitted classifier; the largest engine WDL probability becomes the prediction.
- **Position-only HGB:** histogram gradient boosting using `score_cp`, `score_mate`, `wdl_wins`, `wdl_draws`, `ply`, `material_total`, and `material_diff`.
- **Human-context HGB (main):** the same histogram gradient boosting specification and position features, plus pre-match ratings, rated status, and time-control features.

Percentage game progress is used only to select retrospective test positions. Calculating percentage requires the eventual final game length, which is unknown during prediction and would leak future information. The models therefore receive current `ply` only.

The primary in-game metrics pool all available test snapshots at 10%, 25%, 50%, 75%, and 100% progress; they do not describe one checkpoint or a simple mean of five checkpoint scores. A separate pre-match HGB supplies only the initial 0% estimate. The final model after moves become available is the Human-context HGB.

## Main results

The cleaned dataset contains 19,113 unique games. Across all in-game test snapshots:

| Predictor | Accuracy | Macro-F1 | Log loss |
|---|---:|---:|---:|
| Direct Stockfish WDL | 47.6% | 43.4% | 5.751 |
| Position-only HGB | 63.4% | 52.0% | 0.853 |
| Human-context HGB (main) | **68.7%** | **54.5%** | **0.714** |

Adding human context to the otherwise identical learned model improves accuracy by 5.4 percentage points, macro-F1 by 2.5 points, and log loss by 0.139. The main model progresses from 52.3% accuracy before play to 89.9% at the final recorded position. Direct WDL is especially weak early because Stockfish assigns substantial draw probability to balanced positions, whereas draws are rare in this human-game sample.

## Repository contents

- `CA6001_chess_outcome_prediction.ipynb` — complete, executed analysis and all experiments.
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
