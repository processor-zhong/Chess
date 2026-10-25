"""Standalone traditional models for the feature notebook. No progress inputs.

Uses games/snapshots already prepared by the notebook. No private checkout,
teacher, auxiliary target, or PyTorch dependency is required by this module.
"""
from pathlib import Path
import hashlib
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import f1_score,log_loss,roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

PREMATCH_FEATURES=['white_rating','black_rating','rating_diff','rating_mean','rating_gap','rated_int',
    'base_minutes','increment_seconds','log_base_minutes','white_elo_expectation']
POSITION_FEATURES=['pawn_diff','knight_diff','bishop_diff','rook_diff','queen_diff','material_balance','bishop_pair_diff',
    'mobility_diff','center_control_diff','king_shield_diff','doubled_pawn_diff','isolated_pawn_diff','passed_pawn_diff',
    'castling_rights_diff','white_in_check','black_in_check','white_to_move']
FEATURE_COLUMNS=PREMATCH_FEATURES+POSITION_FEATURES
FAMILIES=['dummy','logistic','gaussian_nb','decision_tree','random_forest','gradient_boosting']
CHECKPOINTS=(.10,.25,.50,.75,1.)
LABELS=['black','draw','white']


def reject_progress(frame):
    forbidden={'progress','actual_progress'}&set(frame.columns)
    if forbidden:raise ValueError('Remove future progress columns before integration: '+', '.join(sorted(forbidden)))


def assemble_inputs(games,snapshots):
    """Join known pre-match values to board rows; return exactly the27 predictors."""
    reject_progress(games);reject_progress(snapshots)
    if not games.id.is_unique:raise ValueError('Deduplicate games before generating snapshots')
    if not np.array_equal(games.loc[snapshots.game_index,'id'].to_numpy(),snapshots.game_id.to_numpy()):
        raise ValueError('Snapshot game_index/game_id alignment')
    pre=games.loc[snapshots.game_index,PREMATCH_FEATURES].reset_index(drop=True)
    X=pd.concat([pre,snapshots[POSITION_FEATURES].reset_index(drop=True)],axis=1)
    if list(X.columns)!=FEATURE_COLUMNS or not np.isfinite(X.to_numpy(dtype=float)).all():raise ValueError('27-column feature contract')
    return X


def make_classifier(family,weighting='none'):
    """Fixed inherited recipes, not newly tuned on this test set."""
    if family not in FAMILIES or weighting not in ['none','balanced']:raise ValueError('Unknown family or weighting')
    weight=None if weighting=='none' else 'balanced'
    if family=='dummy':
        if weighting!='none':raise ValueError('Balanced is not applicable to the empirical-prior reference')
        return DummyClassifier(strategy='prior')
    if family=='logistic':model=LogisticRegression(C=.1,max_iter=3000,class_weight=weight,random_state=42)
    elif family=='gaussian_nb':model=GaussianNB(var_smoothing=.1)
    elif family=='decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        model=DecisionTreeClassifier(max_depth=3,min_samples_leaf=5,class_weight=weight,random_state=42)
    elif family=='random_forest':model=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=10,
        max_features='sqrt',class_weight=weight,random_state=42,n_jobs=4)
    else:model=HistGradientBoostingClassifier(max_iter=150,learning_rate=.1,max_leaf_nodes=7,
        l2_regularization=0.,early_stopping=False,class_weight=weight,random_state=42)
    steps=[('imputer',SimpleImputer(strategy='median')),('nonconstant',VarianceThreshold())]
    if family in ['logistic','gaussian_nb']:steps.append(('scaler',StandardScaler()))
    return Pipeline(steps+[('model',model)])


def fit_classifier(family,weighting,X,y):
    model=make_classifier(family,weighting)
    kwargs={'model__sample_weight':compute_sample_weight('balanced',y)} if family=='gaussian_nb' and weighting=='balanced' else {}
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(X,y,**kwargs)
    return model


def predict_probabilities(model,X):
    reject_progress(X)
    if list(X.columns)!=FEATURE_COLUMNS:raise ValueError('Predict with exactly FEATURE_COLUMNS, in order')
    order=[list(model.classes_).index(i) for i in range(3)]
    return model.predict_proba(X)[:,order]


def metrics(y,p):
    pred=p.argmax(axis=1)
    cr=classification_report(y,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
    draw=cr['draw']
    return dict(n=len(y),accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,pred),log_loss=log_loss(y,p,labels=[0,1,2]),
        macro_auc=roc_auc_score(y,p,labels=[0,1,2],multi_class='ovr',average='macro'),
        draw_precision=draw['precision'],draw_recall=draw['recall'],draw_f1=draw['f1-score'],
        draw_support=int((y==1).sum()),draw_correct=int(((y==1)&(pred==1)).sum()),draw_predicted=int((pred==1).sum()))


def compare_baselines(games,snapshots,output_dir=None):
    """Return(model_registry,metrics_frame); never overwrite existing frozen outputs.

    Fits train only, reports both weight settings on validation and test. Does
    not replace best_handcrafted/best_position/stacker in the original notebook.
    """
    X=assemble_inputs(games,snapshots);y=snapshots.target.to_numpy()
    masks={s:snapshots['split'].eq(s).to_numpy() for s in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if not set(snapshots.loc[masks[a],'game_id']).isdisjoint(set(snapshots.loc[masks[b],'game_id'])):
            raise ValueError('Split by game before generating snapshots')
    out=None
    if output_dir is not None:
        out=Path(output_dir)
        if out.exists():raise FileExistsError('Choose a new result directory: '+str(out))
        for name in ['tables','models','artifacts']:(out/name).mkdir(parents=True)
        stored=pd.concat([snapshots[['game_id','split','target','checkpoint']].reset_index(drop=True),X],axis=1)
        stored.to_csv(out/'tables/inputs.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    registry={};rows=[];perclass=[];params=[]
    with threadpool_limits(limits=4):
        for family in FAMILIES:
            for weighting in (['none'] if family=='dummy' else ['none','balanced']):
                slug=family+'_'+weighting;start=time.time()
                model=fit_classifier(family,weighting,X.loc[masks['train']],y[masks['train']]);registry[slug]=model
                estimator=model if family=='dummy' else model.named_steps['model']
                params.append(dict(slug=slug,family=family,weighting=weighting,parameters=estimator.get_params(),fit_seconds=time.time()-start))
                if out is not None:joblib.dump(model,out/f'models/{slug}.joblib',compress=3)
                for split in ['validation','test']:
                    mask=masks[split];ss=snapshots.loc[mask];p=predict_probabilities(model,X.loc[mask])
                    for checkpoint in ['all',*CHECKPOINTS]:
                        take=np.ones(len(ss),dtype=bool) if checkpoint=='all' else ss.checkpoint.eq(checkpoint).to_numpy()
                        rows.append(dict(slug=slug,family=family,weighting=weighting,split=split,checkpoint=checkpoint,**metrics(y[mask][take],p[take])))
                    if out is not None:
                        table=ss[['game_id','checkpoint','target']].reset_index(drop=True).copy();table['predicted']=p.argmax(axis=1)
                        for i,c in enumerate(LABELS):table['prob_'+c]=p[:,i]
                        table.to_csv(out/f'tables/{slug}_{split}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                        pd.DataFrame(confusion_matrix(y[mask],p.argmax(axis=1),labels=[0,1,2]),index=LABELS,columns=LABELS).to_csv(out/f'tables/{slug}_{split}_cm.csv')
                    cr=classification_report(y[mask],p.argmax(axis=1),labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
                    for label in LABELS:perclass.append(dict(slug=slug,split=split,label=label,**cr[label]))
                print(f'{slug}: completed',flush=True)
    frame=pd.DataFrame(rows)
    if out is not None:
        frame.to_csv(out/'tables/metrics.csv',index=False);pd.DataFrame(perclass).to_csv(out/'tables/per_class.csv',index=False)
        def save(value,path):path.write_bytes((json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n').encode('utf-8'))
        save(params,out/'artifacts/parameters.json')
        save(dict(input_columns=FEATURE_COLUMNS,excluded=['progress','actual_progress','target','checkpoint','ply','ID'],
            python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            notebook_sha256=hashlib.sha256((Path(__file__).parent/'CA6001_chess_outcome_prediction.ipynb').read_bytes()).hexdigest(),
            split='Game-level70/15/15 stratified, seed6001',model_seed=42,
            weights='Train-only N/(3*n_class); NB uses sample_weight, others class_weight',
            source='Fixed recipes from feat/baseline-model-comparison@5a7703c; no new tuning',
            caveats=['Retrospective percentage sampling includes final positions','Historical test already examined, exploratory results']),out/'artifacts/protocol.json')
        save([dict(path=f.relative_to(out).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(out.rglob('*')) if f.is_file()],out/'artifacts/output_hashes.json')
    return registry,frame
