"""Random Forest only: frozen development and one final holdout evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import time
import platform
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix, log_loss)
from sklearn.inspection import permutation_importance
from .plan_a20_interface import load_split, make_model, chronological_cv, contract
from .chess_data import LABELS, sha256

FEATURE_SETS = ['pregame','combined']


def now(): return datetime.now(timezone.utc).isoformat()

def save(value,path):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def folders(out):
    for f in ['tables','figures','models','artifacts']: (out/f).mkdir(parents=True,exist_ok=True)


def metrics(y,pred,proba=None):
    result={'accuracy':accuracy_score(y,pred),
            'macro_f1':f1_score(y,pred,labels=LABELS,average='macro',zero_division=0),
            'balanced_accuracy':balanced_accuracy_score(y,pred)}
    report=classification_report(y,pred,labels=LABELS,output_dict=True,zero_division=0)
    result['draw_recall']=report['draw']['recall']
    if proba is not None: result['log_loss']=log_loss(y,proba,labels=LABELS)
    return result,report,confusion_matrix(y,pred,labels=LABELS)


def score_model(model,X,y):
    pred=model.predict(X)
    order=[list(model.classes_).index(label) for label in LABELS]
    proba=model.predict_proba(X)[:,order]
    m,report,cm=metrics(y,pred,proba)
    return m,report,cm,pred,proba


def export_eval(model,X,y,meta,tag,out):
    m,report,cm,pred,proba=score_model(model,X,y)
    pd.DataFrame(report).T.to_csv(out/'tables'/f'{tag}_per_class.csv')
    pd.DataFrame(cm,index=pd.Index(LABELS,name='actual'),columns=pd.Index(LABELS,name='predicted')).to_csv(out/'tables'/f'{tag}_confusion.csv')
    result=meta[['id','created_at_utc']].copy()
    result['actual']=y.to_numpy();result['predicted']=pred
    for i,label in enumerate(LABELS):result[f'prob_{label}']=proba[:,i]
    result.to_csv(out/'tables'/f'{tag}_predictions.csv',index=False)
    return m,report,cm


def development_hashes(root):
    paths=[root/'artifacts/plan_a20/feature_contract.json',
           root/'data/splits/split_manifest.csv']
    for s in ['train','validation']:
        for prefix in ['X','y','meta']:paths.append(root/f'data/plan_a20/splits/{prefix}_{s}.csv')
    return {str(f.relative_to(root)):sha256(f) for f in paths}


def check_hashes(root,expected):
    for f,h in expected.items():
        if sha256(root/f)!=h: raise ValueError(f'Frozen development input changed: {f}')


def grid():
    # Identical, limited search budget for each feature set. Shared selected parameters.
    return [dict(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,
                 max_features='sqrt',class_weight=weight,n_jobs=4)
            for depth in [8,16,None] for leaf in [3,10] for weight in [None,'balanced']]


def tune(root,out):
    if (out/'artifacts/protocol.json').exists():
        raise ValueError('Run already started. Use a new output directory to avoid overwriting experiments.')
    folders(out)
    protocol={
      'created_utc':now(),'algorithm':'RandomForestClassifier only','seed':42,
      'prediction_ply':20,'full_moves_observed':10,'followup_after_previous_holdout_seen':True,'feature_sets':FEATURE_SETS,'label_order':LABELS,
      'selection_metric':'macro_f1',
      'search_plan':'12 shared configs; 3 expanding chronological CV folds on training; equal mean Macro-F1 across feature sets',
      'shortlist_rule':'Top 3 configs by average CV Macro-F1; ties use candidate ID',
      'freeze_rule':'Highest average validation Macro-F1 across both feature sets; ties use CV score then ID',
      'shared_parameters':True,
      'final_fit_policy':'Refit both fixed configurations on train+validation; fit all preprocessing again',
      'prediction_rule':'Argmax class probability; no probability threshold tuning',
      'importance_policy':'Permutation importance on validation using train-only fitted combined model; never test-based feature selection',
      'test_policy':'No test files loaded in tune; score frozen models once; same historical holdout period was evaluated at ply10, so this is exploratory follow-up, not a new untouched test',
      'development_hashes':development_hashes(root),'candidates':grid(),
      'source_code_sha256':sha256(Path(__file__)),
      'python':platform.python_version(),
      'packages':{x:importlib.metadata.version(x) for x in ['scikit-learn','pandas','numpy','scipy','joblib','matplotlib','chess']}}
    save(protocol,out/'artifacts/protocol.json')
    data={fs:load_split(root,'train',feature_set=fs) for fs in FEATURE_SETS}
    val={fs:load_split(root,'validation',feature_set=fs) for fs in FEATURE_SETS}
    assert data['pregame'][2].id.equals(data['combined'][2].id)
    assert val['pregame'][2].id.equals(val['combined'][2].id)
    folds=list(chronological_cv(data['combined'][2]))
    pd.DataFrame([{'fold':i+1,'train_rows':len(a),'validation_rows':len(b)} for i,(a,b) in enumerate(folds)]).to_csv(out/'tables/cv_folds.csv',index=False)
    rows=[];summaries=[]
    for ci,params in enumerate(grid()):
        for fs in FEATURE_SETS:
            X,y,meta=data[fs]
            values=[]
            for fi,(tr,va) in enumerate(folds,1):
                model=make_model('random_forest',**params)
                tick=time.perf_counter();model.fit(X.iloc[tr],y.iloc[tr]);seconds=time.perf_counter()-tick
                m,_,_,_,_=score_model(model,X.iloc[va],y.iloc[va])
                rows.append({'candidate':ci,'feature_set':fs,'fold':fi,'fit_seconds':seconds,**m})
                values.append(m['macro_f1'])
            summaries.append({'candidate':ci,'feature_set':fs,'mean_cv_macro_f1':float(np.mean(values)),
                              'std_cv_macro_f1':float(np.std(values)), 'parameters':json.dumps(params)})
        pd.DataFrame(rows).to_csv(out/'tables/cv_fold_metrics.csv',index=False)
        print(f'CV candidate {ci+1}/12 complete',flush=True)
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'tables/cv_summary.csv',index=False)
    ranks=summary.groupby('candidate',as_index=False).mean_cv_macro_f1.mean().sort_values(['mean_cv_macro_f1','candidate'],ascending=[False,True])
    shortlist=ranks.head(3).candidate.astype(int).tolist()
    validation=[]
    for ci in shortlist:
        for fs in FEATURE_SETS:
            X,y,_=data[fs];Xv,yv,mv=val[fs]
            model=make_model('random_forest',**grid()[ci])
            tick=time.perf_counter();model.fit(X,y);seconds=time.perf_counter()-tick
            m,_,_,_,_=score_model(model,Xv,yv)
            trainm,_,_,_,_=score_model(model,X,y)
            validation.append({'candidate':ci,'feature_set':fs,'fit_seconds':seconds,
                               'train_accuracy':trainm['accuracy'],'train_macro_f1':trainm['macro_f1'],**m})
    v=pd.DataFrame(validation);v.to_csv(out/'tables/validation_shortlist.csv',index=False)
    ranked=v.groupby('candidate',as_index=False).macro_f1.mean().merge(ranks,on='candidate').sort_values(
        ['macro_f1','mean_cv_macro_f1','candidate'],ascending=[False,False,True])
    ci=int(ranked.iloc[0].candidate)
    selected_rows=[]
    # Save train-only models for interpretation; these are not the final refit models.
    for fs in FEATURE_SETS:
        X,y,_=data[fs];Xv,yv,mv=val[fs]
        model=make_model('random_forest',**grid()[ci]);model.fit(X,y)
        joblib.dump(model,out/'models'/f'rf_{fs}_train_only.joblib')
        m,_,_=export_eval(model,Xv,yv,mv,f'validation_{fs}',out)
        selected_rows.append({'feature_set':fs,**m})
    pd.DataFrame(selected_rows).to_csv(out/'tables/validation_selected_metrics.csv',index=False)
    preferred=sorted(selected_rows,key=lambda r:(-r['macro_f1'],FEATURE_SETS.index(r['feature_set'])))[0]['feature_set']
    # Freeze choice BEFORE computing test scores. Importance cannot change this choice.
    frozen={'frozen_utc':now(),'candidate':ci,'parameters':grid()[ci],
            'preferred_feature_set_by_validation':preferred,
            'final_fit_policy':protocol['final_fit_policy'],
            'prediction_rule':protocol['prediction_rule'],
            'development_hashes':protocol['development_hashes'],
            'protocol_sha256':sha256(out/'artifacts/protocol.json'),
            'selection_table':selected_rows,'test_seen_during_selection':False}
    save(frozen,out/'artifacts/frozen_config.json')
    model=joblib.load(out/'models/rf_combined_train_only.joblib')
    Xv,yv,_=val['combined']
    imp=permutation_importance(model,Xv,yv,scoring='f1_macro',n_repeats=5,random_state=42,n_jobs=1)
    pd.DataFrame({'feature':Xv.columns,'mean_macro_f1_decrease':imp.importances_mean,
                  'std_decrease':imp.importances_std}).sort_values('mean_macro_f1_decrease',ascending=False).to_csv(out/'tables/validation_permutation_importance.csv',index=False)
    save({'completed_utc':now(),'stage':'development_complete','test_loaded':False,
          'frozen_config_sha256':sha256(out/'artifacts/frozen_config.json')},out/'artifacts/development_complete.json')
    print('Frozen configuration:',json.dumps(frozen,ensure_ascii=False),flush=True)


def draw_outputs(out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'artifacts/matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'figure.dpi':140,'savefig.dpi':180})
    for fs in FEATURE_SETS:
        cm=pd.read_csv(out/'tables'/f'test_{fs}_confusion.csv',index_col=0).to_numpy()
        fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
        for ax,values,title in zip(axes,[cm,cm/cm.sum(axis=1,keepdims=True)],['Count','Row-normalized recall']):
            ax.imshow(values,cmap='Blues',vmin=0,vmax=None if title=='Count' else 1)
            for i in range(3):
                for j in range(3):
                    text=str(int(values[i,j])) if title=='Count' else f'{values[i,j]:.1%}'
                    ax.text(j,i,text,ha='center',va='center',color='white' if values[i,j]>values.max()*.55 else 'black')
            ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,
                   xlabel='Predicted',ylabel='Actual',title=title)
        fig.suptitle(f'Random Forest — {fs} — after 10 full moves')
        fig.savefig(out/'figures'/f'test_confusion_{fs}.png');plt.close(fig)
    results=pd.read_csv(out/'tables/test_metrics.csv').set_index('feature_set')
    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,col,title in zip(axes,['accuracy','macro_f1','draw_recall'],['Accuracy','Macro-F1','Draw recall']):
        vals=results.loc[FEATURE_SETS,col]
        bars=ax.bar(['Pre-game','+ Opening'],vals,color=['#52758C','#20776F'])
        ax.set_ylim(0,1);ax.set_title(title)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3)
    fig.suptitle('Same Random Forest parameters; different information sets')
    fig.savefig(out/'figures/test_feature_comparison.png');plt.close(fig)
    importance=pd.read_csv(out/'tables/validation_permutation_importance.csv').head(15).iloc[::-1]
    fig,ax=plt.subplots(figsize=(9,6.5),layout='constrained')
    ax.barh(importance.feature,importance.mean_macro_f1_decrease,color='#20776F')
    ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Mean decrease in validation Macro-F1 (5 permutations)')
    ax.set_title('Validation permutation importance; train-only combined model')
    fig.savefig(out/'figures/validation_feature_importance.png');plt.close(fig)


def finalize(root,out):
    if (out/'artifacts/final_evaluation_complete.json').exists():
        raise ValueError('Final evaluation already completed; inspect saved results rather than rescore.')
    frozen_path=out/'artifacts/frozen_config.json'
    frozen=json.loads(frozen_path.read_text())
    complete=json.loads((out/'artifacts/development_complete.json').read_text())
    assert sha256(frozen_path)==complete['frozen_config_sha256']
    assert sha256(out/'artifacts/protocol.json')==frozen['protocol_sha256']
    check_hashes(root,frozen['development_hashes'])
    # Refit policy is fixed in protocol and config before any holdout loading.
    fitted={};training_players=None
    for fs in FEATURE_SETS:
        Xt,yt,mt=load_split(root,'train',feature_set=fs)
        Xv,yv,mv=load_split(root,'validation',feature_set=fs)
        X=pd.concat([Xt,Xv],ignore_index=True);y=pd.concat([yt,yv],ignore_index=True)
        model=make_model('random_forest',**frozen['parameters'])
        tick=time.perf_counter();model.fit(X,y);seconds=time.perf_counter()-tick
        fitted[fs]=(model,seconds)
        joblib.dump(model,out/'models'/f'rf_{fs}_final.joblib')
        m=pd.concat([mt,mv],ignore_index=True)
        training_players=set(m.white_id)|set(m.black_id)
    save({'unlocked_utc':now(),'frozen_config_sha256':sha256(frozen_path),
          'decision':'Both frozen models refitted on train+validation; no further selection permitted'},
         out/'artifacts/test_access.json')
    results=[];groups=[];holdout_hashes={};test_ids=None
    for fs in FEATURE_SETS:
        X,y,meta=load_split(root,'test',feature_set=fs,allow_test=True)
        if test_ids is None:test_ids=meta.id.copy()
        else:assert test_ids.equals(meta.id)
        model,seconds=fitted[fs]
        m,_,_=export_eval(model,X,y,meta,f'test_{fs}',out)
        results.append({'feature_set':fs,'test_rows':len(X),'refit_rows':len(Xt)+len(Xv),
                        'refit_seconds':seconds,**m})
        seen=meta.white_id.isin(training_players)|meta.black_id.isin(training_players)
        masks={'rating_gap_le100':X.abs_rating_diff.le(100),'rating_gap_gt100':X.abs_rating_diff.gt(100),
               'any_player_seen_in_final_fit':seen,'neither_player_seen_in_final_fit':~seen}
        for name,mask in masks.items():
            if not mask.any():continue
            gm,_,_,_,_=score_model(model,X.loc[mask],y.loc[mask])
            groups.append({'feature_set':fs,'group':name,'rows':int(mask.sum()),
                           'draw_support':int(y.loc[mask].eq('draw').sum()),**gm})
    pd.DataFrame(results).to_csv(out/'tables/test_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'tables/test_subgroup_metrics.csv',index=False)
    r=pd.DataFrame(results).set_index('feature_set')
    delta={m:float(r.loc['combined',m]-r.loc['pregame',m]) for m in ['accuracy','macro_f1','balanced_accuracy','draw_recall','log_loss']}
    save({'comparison':'combined minus pregame, same RF hyperparameters and same test IDs',
          'deltas':delta,'interpretation':'Descriptive differences only; no significance claim; repeated players create dependent samples'},out/'artifacts/feature_effect.json')
    for f in (root/'data/plan_a20/holdout').glob('*.csv'):holdout_hashes[str(f.relative_to(root))]=sha256(f)
    save({'completed_utc':now(),'test_performance_evaluated':True,'test_rows':len(test_ids),
          'frozen_config_sha256':sha256(frozen_path),'holdout_hashes':holdout_hashes,
          'no_parameters_changed_after_test':True,'results':results},out/'artifacts/final_evaluation_complete.json')
    draw_outputs(out)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print('Feature differences:',delta,flush=True)


def main(root):
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['tune','finalize'])
    parser.add_argument('--output',default='modelling_rf/run_20261025_20ply_01')
    args=parser.parse_args()
    out=root/args.output
    if args.stage=='tune':tune(root,out)
    else:finalize(root,out)
