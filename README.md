# Progressive Chess Outcome Prediction

This CA6001 project predicts a chess game's three-class result—Black win, draw, or White win—first from pre-match information and then from board positions reconstructed as the game progresses.

The analysis compares:

- pre-match dummy, logistic-regression, and histogram-gradient-boosting models;
- material-only and richer handcrafted board features;
- a compact convolutional neural network trained on 17 board planes; and
- probability-level fusion of the selected pre-match and position models.

The final evaluation is a pure-prediction curve at 0%, 10%, 25%, 50%, 75%, and 100% of each recorded game. The last point is not replaced with the known result.

## Main results

The cleaned dataset contains 19,113 unique games. On the held-out test set, pre-match accuracy is 52.3% with macro-F1 42.7%. The selected handcrafted position model outperforms the CNN. The fused model reaches 81.2% accuracy and 68.2% macro-F1 at the last recorded position.

## Repository contents

- `CA6001_chess_outcome_prediction.ipynb` — complete, executed analysis and all model experiments.
- `docs/CA6001_Final_Report_Chess_Outcome_Prediction.docx` — final English report.
- `docs/CA6001 Assignment Brief.pdf` — assignment brief.
- `figures/` — seven 300-DPI figures exported by the notebook.
- `pyproject.toml`, `uv.lock`, and `.python-version` — reproducible Python 3.11 environment.

The source CSV is downloaded automatically from the public [Chess Game Dataset (Lichess)](https://www.kaggle.com/datasets/datasnaek/chess) when `data/games.csv` is absent. The `data/` directory is intentionally excluded from version control.

## Run with uv

Install [uv](https://docs.astral.sh/uv/), then run the following commands from the repository root:

```bash
uv sync --locked
uv run jupyter lab
```

Open `CA6001_chess_outcome_prediction.ipynb` and select **Run All**.

For a non-interactive execution that overwrites the notebook with fresh outputs:

```bash
uv run jupyter execute \
  --inplace --timeout=-1 CA6001_chess_outcome_prediction.ipynb
```

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

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Reproducibility notes

- Random seed: `6001`.
- Split: stratified 70% training, 15% validation, and 15% test at game level.
