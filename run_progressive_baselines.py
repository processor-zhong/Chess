"""Reproduce the feature-aligned, progress-free traditional model comparison."""
import argparse
from src.progressive_baselines import project_root,run,DEFAULT_OUTPUT
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=DEFAULT_OUTPUT,help='New project-relative run directory; existing results are protected')
    run(project_root(),p.parse_args().output)
