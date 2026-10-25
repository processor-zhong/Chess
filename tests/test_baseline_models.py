from pathlib import Path
import ast
import json
import unittest
import numpy as np
import pandas as pd
from baseline_models import FEATURE_COLUMNS,PREMATCH_FEATURES,POSITION_FEATURES
from baseline_models import assemble_inputs,compare_baselines,fit_classifier

class BaselineIntegration(unittest.TestCase):
    def fixture(self):
        games=pd.DataFrame(np.ones((3,10)),columns=PREMATCH_FEATURES).assign(id=['a','b','c'])
        snap=pd.DataFrame(np.ones((3,17)),columns=POSITION_FEATURES).assign(game_index=[0,1,2],game_id=['a','b','c'],
            split=['train','validation','test'],target=[0,1,2],checkpoint=[.1,.5,1.])
        return games,snap

    def test_future_progress_is_rejected(self):
        games,snap=self.fixture()
        for field in ['progress','actual_progress']:
            with self.assertRaises(ValueError):assemble_inputs(games,snap.assign(**{field:[.1,.5,1.]}))

    def test_join_preserves_rows_and_exact_27_inputs(self):
        games,snap=self.fixture();snap=snap.iloc[[2,0,1]]
        X=assemble_inputs(games,snap)
        self.assertEqual(list(X.columns),FEATURE_COLUMNS)
        self.assertEqual(len(X),3)
        self.assertNotIn('checkpoint',X)

    def test_game_overlap_rejected_before_fitting(self):
        games,snap=self.fixture();snap.loc[2,['game_index','game_id']]=[0,'a']
        with self.assertRaisesRegex(ValueError,'Split by game'):compare_baselines(games,snap)

    def test_nb_balanced_sample_weights_change_training_prior(self):
        X=pd.DataFrame(np.random.default_rng(42).normal(size=(12,27)),columns=FEATURE_COLUMNS)
        y=np.array([0]*6+[1]*2+[2]*4)
        model=fit_classifier('gaussian_nb','balanced',X,y)
        np.testing.assert_allclose(model.named_steps['model'].class_prior_,np.ones(3)/3)

    def test_all_notebook_predictor_paths_have_progress_removed(self):
        path=Path(__file__).resolve().parents[1]/'CA6001_chess_outcome_prediction.ipynb'
        nb=json.loads(path.read_text(encoding='utf-8'))
        classes={};functions={}
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            tree=ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if isinstance(node,ast.arg):self.assertNotIn('progress',node.arg)
                if isinstance(node,ast.Name):self.assertFalse(node.id.startswith('progress'))
                if isinstance(node,ast.ClassDef):classes[node.name]=node
                if isinstance(node,ast.FunctionDef):functions[node.name]=node
        self.assertEqual([v.arg for v in functions['board_features'].args.args],['board'])
        cnn=classes['PositionCNN'];forward=next(v for v in cnn.body if isinstance(v,ast.FunctionDef) and v.name=='forward')
        self.assertEqual([v.arg for v in forward.args.args],['self','board_input'])
        self.assertFalse(any(isinstance(n,ast.Constant) and n.value=='actual_progress' for n in ast.walk(functions['stack_inputs'])))

if __name__=='__main__':unittest.main()
