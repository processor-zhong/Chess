"""Feature-aligned traditional baselines; true progress never enters a model."""
from pathlib import Path
import hashlib
import json
import os
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import chess
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits
from .baseline_models20 import build_model
from .progressive_features import clean_games, board_features, PROGRESS_POINTS

DEFAULT_OUTPUT='modelling_baselines/run_20261026_feature_01'
LABELS=['black','draw','white']
PRE=['white_rating','black_rating','rating_diff','rating_mean','rating_gap','rated_int',
     'base_minutes','increment_seconds','log_base_minutes','white_elo_expectation']
POS=['pawn_diff','knight_diff','bishop_diff','rook_diff','queen_diff','material_balance','bishop_pair_diff',
     'mobility_diff','center_control_diff','king_shield_diff','doubled_pawn_diff','isolated_pawn_diff',
     'passed_pawn_diff','castling_rights_diff','white_in_check','black_in_check','white_to_move']
INPUTS=PRE+POS
FAMILIES=['dummy','logistic','gaussian_nb','decision_tree','random_forest','gradient_boosting']
NAMES=dict(dummy='多数类基线',logistic='逻辑回归',gaussian_nb='高斯朴素贝叶斯',decision_tree='决策树',
           random_forest='随机森林',gradient_boosting='HistGB')
SHORT=dict(dummy='dummy',logistic='lr',gaussian_nb='nb',decision_tree='dt',random_forest='rf',gradient_boosting='hgb')


def project_root():
    p=Path(__file__).absolute().parents[1]
    # Pandas/open on Windows need an extended path in deep course directories.
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(obj,path):
    Path(path).write_bytes((json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n').encode('utf-8'))


def output_path(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Use a new project-relative output path')
    return root/p


def model_inputs(frame):
    result=frame[INPUTS].copy()
    if list(result.columns)!=INPUTS or not np.isfinite(result.to_numpy(dtype=float)).all():raise ValueError('27-feature contract')
    return result


def train_weights(labels):return compute_sample_weight('balanced',np.asarray(labels))


def prepare_data(root):
    folder=root/'data/progressive'
    manifest=root/'artifacts/progressive/data_manifest.json'
    if manifest.exists():
        d=json.loads(manifest.read_text(encoding='utf-8'))
        for p,h in d['sha256'].items():
            if sha(root/p)!=h:raise ValueError('Cached progressive data changed: '+p)
        return pd.read_csv(folder/'snapshots.csv.gz')
    if folder.exists():raise FileExistsError('Partial data directory exists; preserve it and inspect before rebuilding')
    folder.mkdir(parents=True);manifest.parent.mkdir(parents=True,exist_ok=True)
    games,audit=clean_games(pd.read_csv(root/'data/raw/games.csv'))
    assert len(games)==19113 and games.id.is_unique
    tr,hold=train_test_split(np.arange(len(games)),test_size=.30,random_state=6001,stratify=games.target)
    va,te=train_test_split(hold,test_size=.50,random_state=6001,stratify=games.loc[hold,'target'])
    assert [len(tr),len(va),len(te)]==[13379,2867,2867]
    split=np.full(len(games),'',dtype=object)
    for name,idx in [('train',tr),('validation',va),('test',te)]:split[idx]=name
    games['split']=split
    games[['id','split','target']].to_csv(folder/'game_split.csv',index=False)
    audit.to_csv(folder/'cleaning_audit.csv')
    records=[]
    for i,row in games.iterrows():
        tokens=row.moves.split()
        targets={max(1,min(len(tokens),int(round(len(tokens)*f)))):f for f in PROGRESS_POINTS}
        board=chess.Board()
        for ply,san in enumerate(tokens,1):
            board.push_san(san)
            if ply in targets:
                record=dict(game_index=i,game_id=row.id,split=row['split'],target=int(row.target),checkpoint=targets[ply],ply=ply)
                record.update({f:row[f] for f in PRE});record.update(board_features(board));records.append(record)
        if (i+1)%4000==0:print(f'Replayed {i+1:,}/{len(games):,} games',flush=True)
    frame=pd.DataFrame(records)
    assert len(frame)==94725 and not set(frame.columns)&{'progress','actual_progress'}
    model_inputs(frame)
    frame.to_csv(folder/'snapshots.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    contract=dict(upstream_commit='5f86a39',upstream='https://github.com/processor-zhong/Chess/tree/feature',
        input_columns=INPUTS,pregame_columns=PRE,position_columns=POS,label_order=LABELS,
        excluded_model_fields=['progress','actual_progress','target','winner','game_id','game_index','checkpoint','ply','moves'],
        sampling='Retrospective percentages of final SAN length; metadata only, not live fixed-ply evaluation',
        split='Game-level stratified70/15/15, seed6001; same ID never crosses split',
        count_by_split=frame.groupby('split').size().to_dict(),
        count_by_checkpoint={str(f):frame[frame.checkpoint.eq(f)].groupby('split').size().to_dict() for f in PROGRESS_POINTS})
    save_json(contract,root/'artifacts/progressive/feature_contract.json')
    paths=['data/raw/games.csv','data/progressive/game_split.csv','data/progressive/cleaning_audit.csv',
           'data/progressive/snapshots.csv.gz','artifacts/progressive/feature_contract.json']
    save_json(dict(sha256={p:sha(root/p) for p in paths}),manifest)
    # Training uses the serialized data also used by later verification.
    return pd.read_csv(folder/'snapshots.csv.gz')


def parameters(root):
    old=json.loads((root/'modelling_baselines/run_20261025_20ply_01/artifacts/frozen_config.json').read_text(encoding='utf-8'))
    result={r['family']:r['parameters'] for r in old['selected'] if r['feature_set']=='combined'}
    tree=json.loads((root/'modelling_baselines/run_20261025_tree_01/artifacts/frozen_config.json').read_text(encoding='utf-8'))
    result['decision_tree']=next(r['parameters'] for r in tree['selected'] if r['feature_set']=='combined')
    return result


def make_estimator(family,weighting,params):
    if family=='dummy':return DummyClassifier(strategy='prior')
    cfg=dict(params[family])
    if family!='gaussian_nb':cfg['class_weight']=None if weighting=='none' else 'balanced'
    if family=='decision_tree':return Pipeline([('imputer',SimpleImputer(strategy='median')),('nonconstant',VarianceThreshold()),
        ('model',DecisionTreeClassifier(random_state=42,**cfg))])
    return build_model(family,cfg)


def score(y,prob):
    pred=prob.argmax(axis=1)
    report=classification_report(y,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
    draw=report['draw']
    return dict(n=len(y),accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y,pred),log_loss=log_loss(y,prob,labels=[0,1,2]),
        macro_auc=roc_auc_score(y,prob,labels=[0,1,2],multi_class='ovr',average='macro'),
        draw_precision=draw['precision'],draw_recall=draw['recall'],draw_f1=draw['f1-score'],
        draw_support=int((y==1).sum()),draw_correct=int(((y==1)&(pred==1)).sum()),draw_predicted=int((pred==1).sum()))


def run(root,relative=DEFAULT_OUTPUT):
    out=output_path(root,relative)
    if out.exists():raise FileExistsError('Existing frozen run: '+str(out))
    for name in ['tables','models','artifacts','figures']:(out/name).mkdir(parents=True)
    frame=prepare_data(root);params=parameters(root)
    source=['src/progressive_baselines.py','src/progressive_features.py','src/baseline_models20.py','src/plan_a20_interface.py',
        'src/model_interface.py','src/chess_data.py','run_progressive_baselines.py','verify_progressive_baselines.py','requirements.txt',
        'modelling_baselines/run_20261025_20ply_01/artifacts/frozen_config.json',
        'modelling_baselines/run_20261025_tree_01/artifacts/frozen_config.json','artifacts/progressive/data_manifest.json',
        'artifacts/progressive/feature_contract.json','data/progressive/snapshots.csv.gz']
    protocol=dict(upstream_feature_commit='5f86a39',input_columns=INPUTS,parameters=params,
        class_weights='none or balanced=N/(3*n_class), derived only from train labels',
        nb_balanced='Pipeline model__sample_weight; no native class_weight; scaler remains train-fitted and unweighted',
        dummy_balanced='Not applicable: majority/prior reference depends on observed train distribution',
        selection='Fixed inherited recipes; not re-tuned on the feature-branch split; no test-driven selection',
        progress_fix='Removed true progress from all model inputs; explicit 27-column contract, no probability fusion',
        caveats=['Retrospective percentage sampling still uses final length; final positions included',
            'Multiple snapshots of same game correlated; report per checkpoint as well as pooled',
            'Previously viewed historical test and inherited recipes: exploratory, not a fresh blind test'],
        model_seed=42,split_seed=6001,python=platform.python_version(),sklearn=sklearn.__version__,
        numpy=np.__version__,pandas=pd.__version__,source_sha256={p:sha(root/p) for p in source})
    save_json(protocol,out/'artifacts/protocol.json')
    masks={s:frame['split'].eq(s).to_numpy() for s in ['train','validation','test']}
    X=model_inputs(frame);y=frame.target.to_numpy();w=train_weights(y[masks['train']])
    save_json(dict(counts=np.bincount(y[masks['train']],minlength=3).tolist(),
        class_weights=(len(w)/(3*np.bincount(y[masks['train']],minlength=3))).tolist(),
        effective_balanced_counts=np.bincount(y[masks['train']],weights=w,minlength=3).tolist()),out/'artifacts/train_class_weights.json')
    rows=[];perclass=[];fits=[]
    with threadpool_limits(limits=4):
        for family in FAMILIES:
            for weighting in (['none'] if family=='dummy' else ['none','balanced']):
                slug=SHORT[family]+'_'+weighting
                model=make_estimator(family,weighting,params);start=time.time()
                kwargs={'model__sample_weight':w} if family=='gaussian_nb' and weighting=='balanced' else {}
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    model.fit(X.loc[masks['train']],y[masks['train']],**kwargs)
                joblib.dump(model,out/f'models/{slug}.joblib',compress=3)
                fitted=model if family=='dummy' else model.named_steps['model']
                fits.append(dict(slug=slug,family=family,weighting=weighting,fit_seconds=time.time()-start,parameters=fitted.get_params()))
                for split in ['validation','test']:
                    mask=masks[split];ss=frame.loc[mask];prob=model.predict_proba(X.loc[mask])
                    assert list(model.classes_)==[0,1,2]
                    assert np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1)
                    table=ss[['game_id','checkpoint','target']].reset_index(drop=True).copy();table['predicted']=prob.argmax(axis=1)
                    for i,c in enumerate(LABELS):table['prob_'+c]=prob[:,i]
                    table.to_csv(out/f'tables/{slug}_{split}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                    for point in ['all',*PROGRESS_POINTS]:
                        take=np.ones(len(ss),dtype=bool) if point=='all' else ss.checkpoint.eq(point).to_numpy()
                        rows.append(dict(slug=slug,family=family,weighting=weighting,split=split,
                            checkpoint=point,**score(y[mask][take],prob[take])))
                    cm=confusion_matrix(y[mask],prob.argmax(axis=1),labels=[0,1,2])
                    pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(out/f'tables/{slug}_{split}_cm.csv')
                    cr=classification_report(y[mask],prob.argmax(axis=1),labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0)
                    for label in LABELS:perclass.append(dict(slug=slug,split=split,label=label,**cr[label]))
                print(f'{slug}: test Acc={rows[-6]["accuracy"]:.4f}, Macro-F1={rows[-6]["macro_f1"]:.4f}',flush=True)
    result=pd.DataFrame(rows);result.to_csv(out/'tables/metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(out/'tables/per_class.csv',index=False);save_json(fits,out/'artifacts/fitted_parameters.json')
    render_report(out,result)
    checked=verify(root,out,reload_models=True,check_inventory=False)
    save_json(checked,out/'artifacts/verification.json')
    save_json([dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()],out/'artifacts/output_hashes.json')
    print('Completed',relative,'checks',checked['check_count'],flush=True)


def verify(root,out,reload_models=False,check_inventory=True):
    checks=[]
    def check(ok,name):
        if not bool(ok):raise AssertionError(name)
        checks.append(name)
    if check_inventory:
        for item in json.loads((out/'artifacts/output_hashes.json').read_text(encoding='utf-8')):check(sha(out/item['path'])==item['sha256'],item['path'])
    protocol=json.loads((out/'artifacts/protocol.json').read_text(encoding='utf-8'))
    for p,h in protocol['source_sha256'].items():check(sha(root/p)==h,'Source: '+p)
    data=prepare_data(root);X=model_inputs(data);d=pd.read_csv(out/'tables/metrics.csv')
    check(set(d.weighting)=={'none','balanced'},'Separate weight settings')
    check(len(d)==11*2*6,'All11 models, two splits, six scoring groups')
    check(set(protocol['input_columns'])==set(INPUTS) and len(protocol['input_columns'])==27,'No progress in input contract')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        check(set(data.loc[data['split'].eq(a),'game_id']).isdisjoint(set(data.loc[data['split'].eq(b),'game_id'])),'Disjoint games: '+a+'/'+b)
    tr=data['split'].eq('train');weights=train_weights(data.loc[tr,'target'])
    check(np.allclose(np.bincount(data.loc[tr,'target'],weights=weights),len(weights)/3),'Training-only balanced class totals')
    for fit in json.loads((out/'artifacts/fitted_parameters.json').read_text(encoding='utf-8')):
        slug=fit['slug'];model=joblib.load(out/f'models/{slug}.joblib') if reload_models else None
        if model is not None:
            check(list(model.feature_names_in_)==INPUTS,'Saved 27-feature contract: '+slug)
            if fit['family']=='gaussian_nb':
                prior=model.named_steps['model'].class_prior_
                expected=np.ones(3)/3 if fit['weighting']=='balanced' else np.bincount(data.loc[tr,'target'])/tr.sum()
                check(np.allclose(prior,expected),'NB class prior: '+slug)
        for split in ['validation','test']:
            ss=data.loc[data['split'].eq(split)];p=pd.read_csv(out/f'tables/{slug}_{split}.csv.gz')
            prob=p[['prob_'+c for c in LABELS]].to_numpy();truth=p.target.to_numpy()
            check(np.array_equal(p.game_id,ss.game_id) and np.array_equal(truth,ss.target),'Prediction alignment: '+slug+'/'+split)
            check(np.isfinite(prob).all() and (prob>=0).all() and np.allclose(prob.sum(axis=1),1),'Probabilities: '+slug+'/'+split)
            check(np.array_equal(p.predicted,prob.argmax(axis=1)),'Argmax: '+slug+'/'+split)
            for point in ['all',*PROGRESS_POINTS]:
                take=np.ones(len(p),dtype=bool) if point=='all' else np.isclose(p.checkpoint.to_numpy(),point)
                expected=d[(d.slug==slug)&(d['split']==split)&(d.checkpoint==str(point))].iloc[0]
                for key,value in score(truth[take],prob[take]).items():check(np.isclose(value,expected[key],rtol=1e-9,atol=1e-9),slug+'/'+split+'/'+str(point)+'/'+key)
            cm=pd.read_csv(out/f'tables/{slug}_{split}_cm.csv',index_col=0).to_numpy()
            check(np.array_equal(cm,confusion_matrix(truth,p.predicted,labels=[0,1,2])),'Confusion: '+slug+'/'+split)
            if model is not None:check(np.allclose(model.predict_proba(model_inputs(ss)),prob,rtol=1e-9,atol=1e-9),'Model reload: '+slug+'/'+split)
    return dict(status='passed',check_count=len(checks),checks=checks)


def summary_table(frame):
    lines=['| 模型 | Accuracy | Macro-F1 | 和棋Precision | 和棋Recall | 和棋F1 | Log Loss ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for family in FAMILIES:
        block=frame[frame.family.eq(family)]
        if block.empty:continue
        r=block.iloc[0]
        lines.append(f'| {NAMES[family]} | {r.accuracy:.2%} | {r.macro_f1:.4f} | {r.draw_precision:.2%} | '
            f'{r.draw_recall:.2%} | {r.draw_f1:.4f} | {r.log_loss:.4f} |')
    return '\n'.join(lines)


def render_report(out,result):
    pooled=result[result['split'].eq('test')&result.checkpoint.eq('all')]
    lines=['# 对齐feature的新特征：移除progress输入后的完整基础模型结果','','源分支：`feature@5f86a39`；采用相同数据清洗、游戏级分层划分及百分比采样。',
        '输入统一为10列赛前＋17列局面，所有模型直接使用27列。删除真实progress/actual_progress输入，不使用CNN或概率融合。',
        '训练13,379盘/66,346局面，验证2,867盘/14,200局面，测试2,867盘/14,179局面；按对局划分后才生成局面。',
        '保持旧实验固定参数配方，未在新数据上重新调参。所有none/balanced设置均报告，不按测试成绩择优隐藏配置。','',
        '## 不加权：测试多时点汇总','',summary_table(pooled[pooled.weighting.eq('none')]),'',
        '## balanced：测试多时点汇总','',summary_table(pooled[pooled.weighting.eq('balanced')]),'',
        'Dummy是经验类别先验参照，balanced不适用；NB通过训练sample_weight实施balanced，其余学习器使用class_weight。',
        '权重只用训练标签计算：w_c=N/(3 n_c)。NB的中位数插补和标准化仍只在训练集无权拟合；加权作用于分类器。','',
        '## 每个时点的全部结果','',
        '| 权重 | 模型 | 时点 | Accuracy | Macro-F1 | 和棋Precision | 和棋Recall | 和棋F1 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in result[result['split'].eq('test')&result.checkpoint.ne('all')].itertuples():
        lines.append(f'| {r.weighting} | {NAMES[r.family]} | {float(r.checkpoint):.0%} | {r.accuracy:.2%} | {r.macro_f1:.4f} | '
            f'{r.draw_precision:.2%} | {r.draw_recall:.2%} | {r.draw_f1:.4f} |')
    lines+=['','## 文件索引','',
        '- [全部验证/测试、分时点数值](tables/metrics.csv)，含Balanced Accuracy、Log Loss与OvR宏平均AUC。',
        '- [逐类别Precision/Recall/F1](tables/per_class.csv)。',
        '- [参数记录](artifacts/fitted_parameters.json)、[训练权重](artifacts/train_class_weights.json)、[协议及来源](artifacts/protocol.json)。',
        '- `tables/*_test.csv.gz`及`*_validation.csv.gz`是逐局面概率，`*_cm.csv`是三分类混淆矩阵；文件名前缀对应参数记录slug。',
        '- `models/`为训练模型，`figures/`为比较与分时点图；[完整性检查](artifacts/verification.json)。','',
        '## 修复范围与结论边界','',
        '修复的是把最终总步数推导出的progress作为预测变量的问题；新特征函数不接受progress，显式输入契约排除所有ID、目标、采样位置和未来字段。',
        '百分比采样本身仍依赖最终棋谱长度，100%包含最后记录局面；这是回顾性实验，不是实时固定开局预测。',
        '历史测试已被探索，参数也源于同源历史数据的旧实验，本轮不是新盲测。不同算法搜索经历/训练预算不等，不能代表最优能力。',
        '同一盘棋通常提供5个局面，汇总样本有关联；请同时阅读分时点结果，不能将总体准确率当成前10步成绩。']
    (out/'RESULTS_ZH.md').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    families=FAMILIES[1:];x=np.arange(len(families))
    for ax,key,title in zip(axes,['accuracy','macro_f1','draw_recall'],['Accuracy','Macro-F1','Draw recall']):
        for w,offset,color in [('none',-.18,'#52758c'),('balanced',.18,'#c77a3d')]:
            block=pooled[pooled.weighting.eq(w)].set_index('family').reindex(families)
            ax.bar(x+offset,block[key],.34,label=w,color=color)
        ax.set(title=title,xticks=x,xticklabels=['LR','GaussianNB','Tree','RF','HistGB'],ylim=(0,1));ax.legend(fontsize=8)
    fig.suptitle('Same27 features, no progress input | Pooled retrospective test snapshots')
    fig.savefig(out/'figures/weights.png',dpi=160);plt.close(fig)
