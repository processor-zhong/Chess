"""Verify saved validation predictions and hashes without accessing test data."""
from pathlib import Path
import argparse
import json
from src.baseline_models20 import DEFAULT_OUTPUT
from src.baseline_report20 import verify_results
from src.chess_data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--reload-models', action='store_true',
                        help='Also reload train-only models and reproduce validation probabilities')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root / args.output
    inventory = json.loads((out / 'artifacts/output_hashes.json').read_text(encoding='utf-8'))
    for entry in inventory:
        if sha256(out / entry['path']) != entry['sha256']:
            raise AssertionError(f'Output changed: {entry["path"]}')
    result = verify_results(root, out, reload_models=args.reload_models)
    print(f'Passed {result["check_count"]} result checks and {len(inventory)} output hashes; no test access.')


if __name__ == '__main__':
    main()
