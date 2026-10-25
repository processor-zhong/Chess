"""Rebuild only the selected 20-ply version without exporting legacy inputs."""
from pathlib import Path
import tempfile
import shutil
from src.chess_data import prepare, sha256
from src.plan_a20_data import build_plan_a20


def main():
    root=Path(__file__).resolve().parent
    expected={p:sha256(root/p) for p in ['data/processed/clean_games_for_audit.csv','data/splits/split_manifest.csv']}
    with tempfile.TemporaryDirectory(prefix='chess-clean-') as directory:
        scratch=Path(directory)
        (scratch/'data/raw').mkdir(parents=True)
        shutil.copy2(root/'data/raw/games.csv',scratch/'data/raw/games.csv')
        prepare(scratch)
        for relative,h in expected.items():
            if sha256(scratch/relative)!=h:
                raise ValueError(f'Rebuild differs from shared snapshot: {relative}. Check pinned environment before changing anything.')
        for relative in expected:
            shutil.copy2(scratch/relative,root/relative)
    result=build_plan_a20(root)
    print('Rebuilt selected 20-ply inputs:',result['splits'])
    print('Source hashes match. No model training or test scoring performed.')

if __name__=='__main__':main()
