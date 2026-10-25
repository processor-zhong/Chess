"""Verify committed artifacts and saved scores, without refitting or rescoring models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, confusion_matrix
from src.chess_data import sha256
from src.progressive_baselines import project_root


def main():
    root=project_root()
    run=root/'modelling_rf/run_20261025_20ply_01'
    count=0
    def check(ok,message):
        nonlocal count
        if not bool(ok):raise AssertionError(message)
        count+=1
    raw=json.loads((root/'data/raw/provenance.json').read_text(encoding='utf-8'))
    check(sha256(root/'data/raw/games.csv')==raw['sha256'],'Raw dataset hash')
    for item in json.loads((root/'artifacts/plan_a20/output_hashes.json').read_text(encoding='utf-8')):
        check(sha256(root/item['path'])==item['sha256'],'Data artifact: '+item['path'])
    frozen=json.loads((run/'artifacts/frozen_config.json').read_text(encoding='utf-8'))
    final=json.loads((run/'artifacts/final_evaluation_complete.json').read_text(encoding='utf-8'))
    check(sha256(run/'artifacts/frozen_config.json')==final['frozen_config_sha256'],'Frozen configuration')
    check(sha256(run/'artifacts/protocol.json')==frozen['protocol_sha256'],'Protocol')
    protocol=json.loads((run/'artifacts/protocol.json').read_text(encoding='utf-8'))
    check(sha256(root/'src/rf_experiment20.py')==protocol['source_code_sha256'],'Selected experiment source matches recorded version')
    for relative,h in frozen['development_hashes'].items():check(sha256(root/relative)==h,'Development source: '+relative)
    for relative,h in final['holdout_hashes'].items():check(sha256(root/relative)==h,'Holdout source: '+relative)
    score=pd.read_csv(run/'tables/test_metrics.csv').set_index('feature_set')
    ids=pd.read_csv(root/'data/plan_a20/holdout/meta_test.csv').id
    y=pd.read_csv(root/'data/plan_a20/holdout/y_test.csv').winner
    for fs in ['pregame','combined']:
        d=pd.read_csv(run/f'tables/test_{fs}_predictions.csv')
        check(d.id.equals(ids) and d.actual.equals(y),'Saved prediction alignment: '+fs)
        check(abs((d.actual==d.predicted).mean()-score.loc[fs,'accuracy'])<1e-12,'Saved Accuracy: '+fs)
        check(abs(f1_score(d.actual,d.predicted,average='macro')-score.loc[fs,'macro_f1'])<1e-12,'Saved Macro-F1: '+fs)
        check(abs(log_loss(d.actual,d[['prob_black','prob_draw','prob_white']],labels=['black','draw','white'])-score.loc[fs,'log_loss'])<1e-12,'Saved Log Loss: '+fs)
        saved=pd.read_csv(run/f'tables/test_{fs}_confusion.csv',index_col=0).to_numpy()
        check(np.array_equal(saved,confusion_matrix(d.actual,d.predicted,labels=['black','draw','white'])),'Saved confusion: '+fs)
    # Inventory covers the handover snapshot; .git, runtime caches and this manifest excluded.
    inventory=root/'artifacts/repository_snapshot.json'
    if inventory.exists():
        for item in json.loads(inventory.read_text(encoding='utf-8')):check(sha256(root/item['path'])==item['sha256'],'Repository file: '+item['path'])
    print(f'Passed {count} snapshot checks. No model refitting or new holdout predictions.')

if __name__=='__main__':main()
