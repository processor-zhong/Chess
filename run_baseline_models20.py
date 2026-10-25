"""Train, tune and validate baseline models; never load the test split."""
import os
from pathlib import Path

# Limit nested OpenMP/BLAS work before importing numerical libraries.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

from src.baseline_models20 import main

if __name__ == '__main__':
    main(Path(__file__).resolve().parent)
