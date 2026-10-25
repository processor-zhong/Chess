"""Export the traditional-model handover; retain optional experiments locally."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import argparse
from src.progressive_baselines import project_root


ADDED_FILES = [
    'Baseline_Model_Comparison.ipynb', 'compare_saved_models20.py',
    'run_baseline_models20.py', 'run_decision_tree20.py',
    'verify_baseline_models20.py', 'verify_decision_tree20.py',
    'build_submission20.py', 'docs/BASELINE_MODELS_ZH.md', 'docs/SUBMISSION_ZH.md',
    'src/baseline_models20.py', 'src/baseline_report20.py', 'tests/test_baseline_models20.py',
    'Progressive_Baselines.ipynb', 'run_progressive_baselines.py', 'verify_progressive_baselines.py',
    'src/progressive_baselines.py', 'src/progressive_features.py', 'tests/test_progressive_baselines.py',
    'docs/PROGRESSIVE_BASELINES_ZH.md', 'docs/SUBMISSION_20PLY_ZH.md',
]
RESULT_DIRS = [
    'modelling_baselines/run_20261025_20ply_01',
    'modelling_baselines/run_20261025_tree_01',
    'modelling_baselines/comparison_20ply',
    'modelling_baselines/run_20261026_feature_01',
    'data/progressive', 'artifacts/progressive',
]


def submission_paths(root):
    # Fixed upstream commit contains data and the original RF handover.
    if (root / '.git').exists():
        git_root = str(root)[4:] if str(root).startswith('\\\\?\\') else str(root)
        result = subprocess.run(['git', '-C', git_root, 'ls-tree', '-r', '--name-only', '-z', '0bf217e'],
                                capture_output=True, check=True)
        paths = set(result.stdout.decode('utf-8').strip('\0').split('\0'))
    else:
        # The exported zip has no Git metadata; its audited inventory supplies the list.
        inventory = json.loads((root / 'artifacts/repository_snapshot.json').read_text(encoding='utf-8'))
        paths = {entry['path'] for entry in inventory}
        paths.add('artifacts/repository_snapshot.json')
    paths.update(ADDED_FILES)
    for directory in RESULT_DIRS:
        paths.update(p.relative_to(root).as_posix() for p in (root / directory).rglob('*') if p.is_file())
    for path in paths:
        if not (root / path).is_file():
            raise FileNotFoundError(path)
    return sorted(paths)


def refresh_snapshot(root, paths):
    inventory = []
    for relative in paths:
        if relative == 'artifacts/repository_snapshot.json':
            continue
        data = (root / relative).read_bytes()
        inventory.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (root / 'artifacts/repository_snapshot.json').write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None, help='New zip path; existing files are never overwritten')
    parser.add_argument('--refresh-snapshot-only', action='store_true')
    args = parser.parse_args()
    root = project_root()
    paths = submission_paths(root)
    refresh_snapshot(root, paths)
    if args.refresh_snapshot_only:
        print(f'Refreshed traditional-model inventory: {len(paths)} files')
        return
    output = Path(args.output).resolve() if args.output else root.parent / 'CA6001_Basic_Models_Submission.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for relative in paths:
            z.write(root / relative, 'CA6001_Basic_Models/' + relative)
    print(f'Exported {len(paths)} files, {output.stat().st_size / 1024**2:.1f} MiB: {output}')
    print('Neural experiments, environments, caches and .git are excluded.')


if __name__ == '__main__':
    main()
