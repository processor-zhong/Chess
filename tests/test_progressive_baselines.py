import inspect
import unittest
import numpy as np
import pandas as pd
import chess
from src.progressive_features import board_features
from src.progressive_baselines import INPUTS,model_inputs,train_weights,output_path
from pathlib import Path

class ProgressiveContract(unittest.TestCase):
    def test_board_feature_function_has_no_future_argument(self):
        self.assertEqual(list(inspect.signature(board_features).parameters),['board'])
        values=board_features(chess.Board())
        self.assertEqual(len(values),17)
        self.assertNotIn('progress',values)

    def test_future_metadata_cannot_change_model_inputs(self):
        a=pd.DataFrame(np.ones((3,27)),columns=INPUTS)
        b=a.assign(progress=[.1,.5,1.],actual_progress=[1.,.5,.1],target=[0,1,2],checkpoint=[.1,.5,1.])
        pd.testing.assert_frame_equal(model_inputs(a),model_inputs(b))
        self.assertEqual(len(model_inputs(b).columns),27)

    def test_balanced_weights_use_each_class_training_count(self):
        y=np.array([0,0,0,1,2,2]);w=train_weights(y)
        np.testing.assert_allclose(np.bincount(y,weights=w),[2,2,2])
        self.assertGreater(w[y==1][0],w[y==0][0])

    def test_output_path_cannot_escape_project(self):
        with self.assertRaises(ValueError):output_path(Path.cwd(),'../overwrite')
        with self.assertRaises(ValueError):output_path(Path.cwd(),str(Path.cwd().absolute()))

if __name__=='__main__':unittest.main()
