"""Independent structural, prefix and API checks. Never score test models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.chess_data import LABELS, sha256
from src.plan_a20_data import PLY, opening_features, dump_json
from src.plan_a20_interface import contract, load_split, make_model, chronological_cv


def validate(root):
    root = Path(root)
    c = contract(root)
    checks=[]
    def check(ok, name):
        if not bool(ok): raise AssertionError(name)
        checks.append(name)
    clean_path=root/'data/processed/clean_games_for_audit.csv'
    manifest_path=root/'data/splits/split_manifest.csv'
    check(sha256(clean_path)==c['source_clean_sha256'], 'Original clean snapshot unchanged')
    check(sha256(manifest_path)==c['source_manifest_sha256'], 'Original split manifest unchanged')
    clean=pd.read_csv(clean_path).set_index('id',drop=False)
    manifest=pd.read_csv(manifest_path).set_index('id')
    audit=pd.read_csv(root/'tables/plan_a20/row_decisions.csv',keep_default_na=False).set_index('id')
    check(audit.index.is_unique and set(audit.index)==set(clean.index), 'Every clean ID has one decision')
    check(audit.original_split.equals(manifest.loc[audit.index,'split'].rename('original_split')), 'Original membership preserved in audit')
    check(set(c['combined_features']).isdisjoint(c['forbidden_inputs']), 'No forbidden field in model contract')
    arrays={}; idsets={}
    for split in ['train','validation','test']:
        X,y,meta=load_split(root,split,allow_test=True)
        pre,py,pm=load_split(root,split,feature_set='pregame',allow_test=True)
        ids=meta.id.tolist();source=clean.loc[ids]
        arrays[split]=(X,y,meta);idsets[split]=set(ids)
        check(meta.id.is_unique and len(X)==len(y)==len(meta), f'{split}: unique IDs and aligned lengths')
        check(idsets[split]==set(audit.index[(audit.original_split==split)&(audit.decision=='retained')]), f'{split}: exact retained cohort')
        check(manifest.loc[ids,'split'].eq(split).all(), f'{split}: no original split reassignment')
        check(source.moves.str.split().str.len().gt(PLY).all(), f'{split}: all games continue after cutoff')
        check(np.array_equal(source.winner.to_numpy(),y.to_numpy()), f'{split}: labels align to original IDs (structure only)')
        check(np.allclose(source[c['pregame_features']].to_numpy(),pre.to_numpy()), f'{split}: pregame values unchanged')
        check(np.isfinite(X.to_numpy()).all(), f'{split}: finite numeric features')
        check(pre.equals(X[c['pregame_features']]) and py.equals(y) and pm.equals(meta), f'{split}: both feature sets have identical rows')
        check(pd.to_datetime(meta.created_at_utc,utc=True,format='ISO8601').is_monotonic_increasing, f'{split}: chronological ordering')
        check(list(X.columns)==c['combined_features'], f'{split}: exact column order')
        check(set(y).issubset(LABELS),f'{split}: valid labels')
        # Exhaustive prefix recomputation, including test features; no outcome statistics.
        rebuilt=[];fens=[]
        for moves in source.moves:
            feat,fen=opening_features(moves.split()[:PLY]);rebuilt.append(feat);fens.append(fen)
        check(np.array_equal(pd.DataFrame(rebuilt)[c['opening_features']].to_numpy(),X[c['opening_features']].to_numpy()), f'{split}: all opening features recompute from only 20 plies')
        check(fens==meta.position_fen_at_ply20.tolist(), f'{split}: audited FENs match prefixes')
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        check(idsets[left].isdisjoint(idsets[right]),f'{left}/{right}: no repeated game IDs')
        lm=arrays[left][2];rm=arrays[right][2]
        check(pd.to_datetime(lm.created_at_utc,utc=True,format='ISO8601').max()<pd.to_datetime(rm.created_at_utc,utc=True,format='ISO8601').min()-pd.Timedelta(days=1), f'{left}/{right}: original temporal gap preserved')
        check(pd.to_datetime(lm.last_move_at_utc,utc=True,format='ISO8601').max()<pd.to_datetime(rm.created_at_utc,utc=True,format='ISO8601').min(), f'{left}/{right}: earlier recorded outcomes precede later starts')
    check(audit.loc[manifest.index[manifest.split=='excluded'],'decision'].eq('original_boundary_gap').all(), 'All original gaps remain excluded')
    short=clean.moves.str.split().str.len().le(PLY)&manifest.loc[clean.index,'split'].ne('excluded')
    check(audit.loc[clean.index[short],'decision'].eq('finished_by_ply20').all(), 'All short active games explicitly excluded')
    prefix='e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7'.split()
    feat,fen=opening_features(prefix)
    check(feat['white_castled']==1 and feat['black_castled']==1 and feat['material_diff']==0,'Known castle/material example correct')
    ep,_=opening_features('e4 a6 e5 d5 exd6 cxd6 Nf3 Nf6 d4 e6 Bd3 Be7 O-O O-O Re1 Nc6 c3 Bd7 Bf4 Rc8'.split())
    check(ep['white_captures_first20']==1 and ep['black_captures_first20']==1 and ep['white_pawn_count']==7 and ep['black_pawn_count']==7, 'En passant capture and piece counts correct')
    # Changes after ply 10 must have no effect on input vectors.
    check(opening_features((prefix+['ILLEGAL_FUTURE_TOKEN'])[:PLY])==(feat,fen), 'Arbitrary future suffix cannot affect features')
    invalid=prefix.copy();invalid[0]='--'
    try: opening_features(invalid)
    except ValueError: checks.append('Null moves rejected')
    else: raise AssertionError('Null move accepted')
    invalid=prefix.copy();invalid[0]='e5'
    try: opening_features(invalid)
    except ValueError: checks.append('Illegal SAN rejected')
    else: raise AssertionError('Illegal SAN accepted')
    try: load_split(root,'test')
    except ValueError: checks.append('Default loader blocks test access')
    else: raise AssertionError('Test guard missing')
    for i,(tr,va) in enumerate(chronological_cv(arrays['train'][2]),1):
        check(not(set(tr)&set(va)) and set(arrays['train'][1].iloc[tr])==set(LABELS) and set(arrays['train'][1].iloc[va])==set(LABELS),f'Train CV {i}: valid disjoint indices and all classes')
    # Smoke test fitted pipelines using training only; no model performance scores.
    smoke=[]
    for fs in ['pregame','combined']:
        X,y,_=load_split(root,'train',feature_set=fs)
        for name in ['logistic','random_forest']:
            model=make_model(name,**({'n_estimators':20,'n_jobs':1} if name=='random_forest' else {}))
            model.fit(X,y)
            proba=model.predict_proba(X.iloc[:5])
            check(proba.shape==(5,3) and np.allclose(proba.sum(axis=1),1), f'{name}/{fs}: fit and probability API works')
            support=model.named_steps['nonconstant'].get_support()
            if fs=='combined':
                check(any(support[list(X.columns).index(f)] for f in c['opening_features']),f'{name}: opening features reach estimator')
            smoke.append({'model':name,'feature_set':fs,'input_features':len(X.columns),'nonconstant_features':int(support.sum()),'scored':False})
    # Validate artifact bytes last; source hash list intentionally excludes validation report.
    for item in json.loads((root/'artifacts/plan_a20/output_hashes.json').read_text()):
        check(sha256(root/item['path'])==item['sha256'],f"Hash: {item['path']}")
    result={'status':'passed','check_count':len(checks),'checks':checks,'pipeline_smoke_tests':smoke,
            'test_performance_evaluated':False,
            'note':'Test labels checked only for source alignment/validity; no test class summaries or scoring.'}
    dump_json(result,root/'artifacts/plan_a20/validation_checks.json')
    return result

if __name__=='__main__':
    result=validate(Path(__file__).resolve().parent)
    print(f"Passed {result['check_count']} Plan A checks; no test scores computed.")
