"""Verify the integrated traditional models; does not require PyTorch or retraining."""
from pathlib import Path
import argparse
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from baseline_models import FEATURE_COLUMNS,LABELS,metrics,predict_probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/baselines_20261026')
    parser.add_argument('--reload-models',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;out=root/args.output;checks=0
    def check(ok,label):
        nonlocal checks
        if not bool(ok):raise AssertionError(label)
        checks+=1
    for item in json.loads((out/'artifacts/output_hashes.json').read_text(encoding='utf-8')):
        check(hashlib.sha256((out/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path'])
    protocol=json.loads((out/'artifacts/protocol.json').read_text(encoding='utf-8'))
    check(hashlib.sha256((root/'baseline_models.py').read_bytes()).hexdigest()==protocol['module_sha256'],'Module provenance')
    check(hashlib.sha256((root/'CA6001_chess_outcome_prediction.ipynb').read_bytes()).hexdigest()==protocol['notebook_sha256'],'Notebook provenance')
    data=pd.read_csv(out/'tables/inputs.csv.gz');scores=pd.read_csv(out/'tables/metrics.csv')
    check(len(scores)==132 and len(data)==94725,'Full input/evaluation counts')
    check(not set(data)&{'progress','actual_progress'},'No progress in stored model input table')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        check(set(data.loc[data['split'].eq(a),'game_id']).isdisjoint(set(data.loc[data['split'].eq(b),'game_id'])),'Disjoint games')
    for row in json.loads((out/'artifacts/parameters.json').read_text(encoding='utf-8')):
        slug=row['slug'];model=joblib.load(out/f'models/{slug}.joblib') if args.reload_models else None
        for split in ['validation','test']:
            ss=data[data['split'].eq(split)];p=pd.read_csv(out/f'tables/{slug}_{split}.csv.gz')
            truth=p.target.to_numpy();prob=p[['prob_'+c for c in LABELS]].to_numpy()
            check(np.array_equal(p.game_id,ss.game_id) and np.array_equal(p.target,ss.target),'Saved alignment')
            check(np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1),'Probability validity')
            check(np.array_equal(prob.argmax(axis=1),p.predicted),'Argmax')
            for point in ['all',.1,.25,.5,.75,1.]:
                mask=np.ones(len(p),dtype=bool) if point=='all' else np.isclose(p.checkpoint,point)
                expected=scores[(scores.slug==slug)&(scores['split']==split)&(scores.checkpoint==str(point))].iloc[0]
                for key,value in metrics(truth[mask],prob[mask]).items():check(np.isclose(value,expected[key],rtol=1e-9,atol=1e-9),slug+'/'+key)
            if model is not None:check(np.allclose(predict_probabilities(model,ss[FEATURE_COLUMNS]),prob,rtol=1e-9,atol=1e-9),'Reloaded model')
    print(f'Passed {checks} integration checks; no refitting.')


if __name__=='__main__':main()
