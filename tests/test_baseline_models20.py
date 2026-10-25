"""Protect selection and output boundaries, rather than asserting model scores."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.baseline_models20 import reserve_output, select_candidate
from src.plan_a20_interface import load_split


class ExperimentBoundaries(unittest.TestCase):
    def test_cv_selection_and_deterministic_ties(self):
        table = pd.DataFrame({'candidate': [3, 2, 1, 0],
                              'cv_macro_f1': [.4, .5, .5, .5],
                              'cv_log_loss': [.1, .7, .6, .6]})
        self.assertEqual(select_candidate(table), 0)
        self.assertEqual(select_candidate(table.sample(frac=1, random_state=42)), 0)

    def test_selection_rejects_validation_column(self):
        table = pd.DataFrame({'candidate': [0], 'cv_macro_f1': [.4], 'cv_log_loss': [.8],
                              'validation_macro_f1': [.9]})
        with self.assertRaises(ValueError):
            select_candidate(table)

    def test_test_guard_precedes_any_file_read(self):
        with patch('src.plan_a20_interface.pd.read_csv') as reader:
            with self.assertRaises(ValueError):
                load_split(Path('nonexistent-checkout'), 'test')
            reader.assert_not_called()

    def test_output_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaises(ValueError):
                reserve_output(root, '../outside')
            with self.assertRaises(ValueError):
                reserve_output(root, '.')
            out = reserve_output(root, 'new-experiment')
            marker = out / 'original.txt'
            marker.write_text('keep me', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                reserve_output(root, 'new-experiment')
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep me')


if __name__ == '__main__':
    unittest.main()
