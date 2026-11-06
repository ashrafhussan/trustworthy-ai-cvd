"""Reproducible pipeline: computed Trustworthiness Score for CVD prediction.
Download processed.{cleveland,hungarian,va}.data from
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/ into this folder, then run:
  pip install scikit-learn xgboost interpret-core shap lime matplotlib scipy pandas
  python trust_cvd_pipeline.py
Stage 1 writes results.json; stage 2 computes the Trust Score (final.json); stage 3 draws figs/. Runtime ~15 min on a laptop."""
import pandas as pd, numpy as np, json, warnings, time
warnings.filterwarnings('ignore')
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import spearmanr, t as tdist
import shap
from lime.lime_tabular import LimeTabularExplainer
RNG=42; np.random.seed(RNG)
cols=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
def load(f):
    d=pd.read_csv(f'processed.{f}.data',header=None,names=cols,na_values='?')
    d.loc[d.chol==0,'chol']=np.nan; d.loc[d.trestbps==0,'trestbps']=np.nan
    y=(d.num>0).astype(int).values; return d.drop(columns='num'),y
Xc,yc=load('cleveland'); Xh,yh=load('hungarian'); Xv,yv=load('va')
FULL=cols[:-1]; RED=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']
CAT=['cp','restecg','slope','thal']; CONT=['age','trestbps','chol','thalach','oldpeak']
def pre(feats):
    cat=[c for c in CAT if c in feats]; num=[c for c in feats if c not in cat]
    return ColumnTransformer([('n',Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]),num),
                              ('c',Pipeline([('i',SimpleImputer(strategy='most_frequent')),('o',OneHotEncoder(handle_unknown='ignore'))]),cat)])
def models():
    return {'LR':LogisticRegression(C=1.0,max_iter=2000),
            'RF':RandomForestClassifier(n_estimators=300,min_samples_leaf=3,random_state=RNG,n_jobs=-1),
            'XGBoost':XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=RNG,verbosity=0,n_jobs=4),
            'FNN':MLPClassifier(hidden_layer_sizes=(16,8),alpha=1e-1,early_stopping=True,validation_fraction=0.15,n_iter_no_change=20,learning_rate_init=0.005,max_iter=1000,random_state=RNG),
            'EBM':ExplainableBoostingClassifier(interactions=5,random_state=RNG,n_jobs=1)}
def pipe(name,feats): return Pipeline([('p',pre(feats)),('m',models()[name])])
NAMES=list(models().keys()); R={}
# ---------- 1. repeated CV on Cleveland (full features) ----------
rskf=RepeatedStratifiedKFold(n_splits=10,n_repeats=5,random_state=RNG)
folds=list(rskf.split(Xc,yc))
cv={n:{'auc':[],'brier':[],'acc':[],'rec':[],'prec':[],'f1':[],'flip':[],'dauc':[]} for n in NAMES}
sd=Xc[CONT].std()
rng=np.random.default_rng(RNG)
t0=time.time()
for k,(tr,te) in enumerate(folds):
    for n in NAMES:
        m=pipe(n,FULL).fit(Xc.iloc[tr],yc[tr]); Xt=Xc.iloc[te]; p=m.predict_proba(Xt)[:,1]; yh_=(p>=.5).astype(int)
        c=cv[n]; c['auc'].append(roc_auc_score(yc[te],p)); c['brier'].append(brier_score_loss(yc[te],p))
        c['acc'].append(accuracy_score(yc[te],yh_)); c['rec'].append(recall_score(yc[te],yh_)); c['prec'].append(precision_score(yc[te],yh_,zero_division=0)); c['f1'].append(f1_score(yc[te],yh_))
        fl=[];da=[]
        for r in range(10):
            Xn=Xt.copy()
            for col in CONT: Xn[col]=Xn[col]+rng.normal(0,0.10*sd[col],len(Xn))
            pn=m.predict_proba(Xn)[:,1]; fl.append(np.mean((pn>=.5)!=(p>=.5))); da.append(roc_auc_score(yc[te],pn)-c['auc'][-1])
        c['flip'].append(np.mean(fl)); c['dauc'].append(np.mean(da))
print('cv done',time.time()-t0)
R['cv']={n:{k:[float(x) for x in v] for k,v in cv[n].items()} for n in NAMES}
# corrected resampled t-test (Nadeau-Bengio) vs LR
def nb(a,b,ntr,nte):
    d=np.array(a)-np.array(b); J=len(d); m=d.mean(); v=d.var(ddof=1)
    tt=m/np.sqrt((1/J+nte/ntr)*v); p=2*tdist.sf(abs(tt),J-1); return float(m),float(tt),float(p)
ntr=len(folds[0][0]); nte=len(folds[0][1])
R['tests']={n:nb(cv[n]['auc'],cv['LR']['auc'],ntr,nte) for n in NAMES if n!='LR'}
# ---------- 2. external validation (reduced features) ----------
ext={}; rngb=np.random.default_rng(1)
for n in NAMES:
    m=pipe(n,RED).fit(Xc[RED],yc); ext[n]={}
    pin=[]; 
    for (tr,te) in folds[:10]:
        mm=pipe(n,RED).fit(Xc[RED].iloc[tr],yc[tr]); pin.append(roc_auc_score(yc[te],mm.predict_proba(Xc[RED].iloc[te])[:,1]))
    ext[n]['int_auc_red']=float(np.mean(pin))
    for nm,(X,y) in {'Hungarian':(Xh,yh),'VA':(Xv,yv)}.items():
        p=m.predict_proba(X[RED])[:,1]
        ext[n][nm]={'auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),'p':p.tolist()}
# bootstrap CIs + paired diff vs LR
for nm,(X,y) in {'Hungarian':(Xh,yh),'VA':(Xv,yv)}.items():
    B=2000; idx=[rngb.integers(0,len(y),len(y)) for _ in range(B)]
    boot={n:[] for n in NAMES}
    for ii in idx:
        if len(set(y[ii]))<2: continue
        for n in NAMES: boot[n].append(roc_auc_score(y[ii],np.array(ext[n][nm]['p'])[ii]))
    for n in NAMES:
        b=np.array(boot[n]); ext[n][nm]['ci']=[float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
        if n!='LR':
            d=b-np.array(boot['LR']); ext[n][nm]['diff_vs_LR']=float(ext[n][nm]['auc']-ext['LR'][nm]['auc']); ext[n][nm]['p_vs_LR']=float(2*min((d<=0).mean(),(d>=0).mean()))
for n in NAMES:
    for nm in ['Hungarian','VA']: ext[n][nm].pop('p')
R['ext']=ext; print('ext done',time.time()-t0)
# ---------- 3. explanations ----------
bg=Xc.sample(50,random_state=1); ev=Xc.sample(60,random_state=2)
def shap_vals(m,X):
    f=lambda A: m.predict_proba(pd.DataFrame(A,columns=FULL))[:,1]
    e=shap.explainers.Permutation(f,shap.maskers.Independent(bg.fillna(bg.median()).values),max_evals=2*len(FULL)+1,seed=0)
    return e(X.fillna(Xc.median()).values).values
prior={'age':1,'sex':1,'cp':1,'trestbps':1,'chol':1,'fbs':1,'restecg':1,'thalach':-1,'exang':1,'oldpeak':1,'slope':1,'ca':1,'thal':1}
def riskcode(X):
    Z=X.fillna(Xc.median()).copy().astype(float)
    Z['cp']=(Z.cp==4).astype(float); Z['restecg']=(Z.restecg>0).astype(float); Z['slope']=(Z.slope>=2).astype(float); Z['thal']=(Z.thal>=6).astype(float)
    return Z
Zev=riskcode(ev)
expl={}; lime_e=LimeTabularExplainer(Xc.fillna(Xc.median()).values,feature_names=FULL,categorical_features=[FULL.index(c) for c in ['sex','cp','fbs','restecg','exang','slope','ca','thal']],discretize_continuous=False,mode='classification',random_state=0)
for n in NAMES:
    m=pipe(n,FULL).fit(Xc,yc); S=shap_vals(m,ev); imp=np.abs(S).mean(0)
    # clinical directional concordance
    conc=[];w=[]
    for j,f in enumerate(FULL):
        r=spearmanr(Zev[f],S[:,j]).correlation
        if np.isnan(r): r=0
        conc.append(1.0 if np.sign(r)==prior[f] else 0.0); w.append(imp[j])
    w=np.array(w); C=float((np.array(conc)*w).sum()/w.sum())
    # SHAP-LIME local agreement
    f=lambda A: m.predict_proba(pd.DataFrame(A,columns=FULL))
    ag=[]
    for i in range(30):
        x=ev.fillna(Xc.median()).values[i]
        le=lime_e.explain_instance(x,f,num_features=13,num_samples=2000)
        lw=np.zeros(13)
        for j,v in le.as_map()[1]: lw[j]=v
        r=spearmanr(np.abs(lw),np.abs(S[i])).correlation; ag.append(0 if np.isnan(r) else r)
    # bootstrap stability
    imps=[]
    for b in range(15):
        ii=np.random.default_rng(100+b).integers(0,len(yc),len(yc))
        mb=pipe(n,FULL).fit(Xc.iloc[ii],yc[ii]); imps.append(np.abs(shap_vals(mb,ev)).mean(0))
    st=[spearmanr(imps[a],imps[b]).correlation for a in range(15) for b in range(a+1,15)]
    top5=list(np.array(FULL)[np.argsort(-imp)[:5]])
    expl[n]={'imp':imp.tolist(),'conc':conc,'C':C,'agree':float(np.mean(ag)),'stab':float(np.mean(st)),'top5':top5}
    print(n,'expl',round(time.time()-t0),expl[n]['C'],expl[n]['agree'],expl[n]['stab'],top5)
R['expl']=expl
json.dump(R,open('results.json','w'))

# ================= Stage 2: Trust Score =================
import json,numpy as np
R=json.load(open('results.json'))
N=['LR','EBM','RF','XGBoost','FNN']
T={}
for n in N:
    c=R['cv'][n]; auc=np.mean(c['auc']); br=np.mean(c['brier'])
    P=0.5*(auc-.5)/.5+0.5*(1-br/.25)
    e=R['expl'][n]; E=np.mean([e['stab'],max(e['agree'],0)])
    tr=np.mean([R['ext'][n][s]['auc'] for s in ['Hungarian','VA']])/R['ext'][n]['int_auc_red']
    Rb=np.mean([1-np.mean(c['flip']),min(1,tr)])
    C=e['C']
    T[n]=dict(P=P,E=E,R=Rb,C=C,transport=tr,flip=np.mean(c['flip']))
W={'Original 30/30/30/10':[.1,.3,.3,.3],'Equal (25 each)':[.25,.25,.25,.25],'Performance-heavy 55/15/15/15':[.55,.15,.15,.15]}
for wn,w in W.items():
    for n in N: T[n][wn]=100*(w[0]*T[n]['P']+w[1]*T[n]['E']+w[2]*T[n]['R']+w[3]*T[n]['C'])
rng=np.random.default_rng(0); D=rng.dirichlet([1,1,1,1],20000)
M=np.array([[T[n]['P'],T[n]['E'],T[n]['R'],T[n]['C']] for n in N])
S=D@M.T; best=S.argmax(1)
top={n:float((best==i).mean()) for i,n in enumerate(N)}
ranks=(-S).argsort(1).argsort(1)+1
mr={n:float(ranks[:,i].mean()) for i,n in enumerate(N)}
# ties within 1 point
for n in N: print(n,{k:round(v,3) for k,v in T[n].items()})
print('top',top,'meanrank',mr)
R['trust']=T; R['sens']={'top':top,'meanrank':mr}
json.dump(R,open('final.json','w'))

# ================= Stage 3: Figures =================
import os; os.makedirs('figs',exist_ok=True)
import json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'serif','font.size':7,'axes.linewidth':0.6})
R=json.load(open('final.json')); N=['LR','EBM','RF','XGBoost','FNN']
FULL=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
col={'LR':'#1f4e79','EBM':'#2e8b57','RF':'#b8860b','XGBoost':'#a23b3b','FNN':'#6a4c93'}
# Fig1 pipeline
fig,ax=plt.subplots(figsize=(3.4,2.3)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,10)
boxes=[(5,9.1,'Cleveland cohort (n=303) — development'),
       (5,7.5,'Pipeline: impute → scale → one-hot (fit in-fold)'),
       (5,5.9,'Five models: LR · EBM · RF · XGBoost · FNN'),
       (2.6,4.1,'10-fold CV × 5 repeats\n(internal performance)'),(7.4,4.1,'External validation\nHungarian (n=294), VA (n=200)'),
       (5,2.3,'Measured components: P · E · R · C'),
       (5,0.7,'Composite Trust Score + weight sensitivity')]
for x,y,t in boxes:
    w=4.4 if x!=5 else 9.2
    ax.add_patch(FancyBboxPatch((x-w/2,y-0.6),w,1.2,boxstyle='round,pad=0.05',fc='#eef3f8',ec='#1f4e79',lw=0.7))
    ax.text(x,y,t,ha='center',va='center',fontsize=6.3)
for (x1,y1,x2,y2) in [(5,8.5,5,8.1),(5,6.9,5,6.5),(4,5.3,2.8,4.7),(6,5.3,7.2,4.7),(2.6,3.5,4.2,2.9),(7.4,3.5,5.8,2.9),(5,1.7,5,1.3)]:
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.7,color='#333'))
plt.tight_layout(pad=0.1); plt.savefig('figs/fig1.png',dpi=300); plt.close()
# Fig2: CV AUC box + external AUC
fig,ax=plt.subplots(1,2,figsize=(3.4,1.75))
bp=ax[0].boxplot([R['cv'][n]['auc'] for n in N],widths=0.55,patch_artist=True,medianprops=dict(color='k',lw=0.8),flierprops=dict(markersize=2),boxprops=dict(lw=0.6),whiskerprops=dict(lw=0.6),capprops=dict(lw=0.6))
for p,n in zip(bp['boxes'],N): p.set_facecolor(col[n]); p.set_alpha(0.6)
ax[0].set_xticks(range(1,6)); ax[0].set_xticklabels(['LR','EBM','RF','XGB','FNN'],fontsize=6); ax[0].set_ylabel('AUC'); ax[0].set_title('(a) Internal CV',fontsize=7)
for j,s in enumerate(['Hungarian','VA']):
    for i,n in enumerate(N):
        e=R['ext'][n][s]; x=i+(j-0.5)*0.3
        ax[1].errorbar(x,e['auc'],yerr=[[e['auc']-e['ci'][0]],[e['ci'][1]-e['auc']]],fmt='o' if j==0 else 's',color=col[n],ms=2.5,lw=0.7,capsize=1.5)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(['LR','EBM','RF','XGB','FNN'],fontsize=6); ax[1].set_title('(b) External: o Hung., □ VA',fontsize=7); ax[1].set_ylim(0.6,0.95)
plt.tight_layout(pad=0.4); plt.savefig('figs/fig2.png',dpi=300); plt.close()
# Fig3: importance heatmap
M=np.array([R['expl'][n]['imp'] for n in N]); M=M/M.sum(1,keepdims=True)
order=np.argsort(-M.mean(0))
fig,ax=plt.subplots(figsize=(3.4,1.5))
im=ax.imshow(M[:,order],cmap='Blues',aspect='auto')
ax.set_xticks(range(13)); ax.set_xticklabels(np.array(FULL)[order],rotation=60,fontsize=5.8); ax.set_yticks(range(5)); ax.set_yticklabels(N,fontsize=6)
for i in range(5):
    for j in range(13):
        v=M[i,order[j]]; ax.text(j,i,f'{v*100:.0f}',ha='center',va='center',fontsize=4.6,color='white' if v>0.15 else 'black')
cb=plt.colorbar(im,ax=ax,fraction=0.03,pad=0.01); cb.ax.tick_params(labelsize=5)
plt.tight_layout(pad=0.2); plt.savefig('figs/fig3.png',dpi=300); plt.close()
# Fig4: components + sensitivity
fig,ax=plt.subplots(1,2,figsize=(3.4,1.95),gridspec_kw={'width_ratios':[2.0,1.2]})
comps=['P','E','R','C']; w=0.15
for i,n in enumerate(N):
    ax[0].bar(np.arange(4)+(i-2)*w,[R['trust'][n][c] for c in comps],w,color=col[n],label=n)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(['P (perf.)','E (expl.)','R (rob.)','C (clin.)'],fontsize=5); ax[0].set_ylim(0.5,1.02); ax[0].set_title('(a) Measured components',fontsize=7)
ax[0].legend(fontsize=5,ncol=5,loc='upper center',bbox_to_anchor=(0.75,-0.13),frameon=False,handlelength=1,columnspacing=0.6)
top=R['sens']['top']; ax[1].barh(range(5)[::-1],[top[n]*100 for n in N],color=[col[n] for n in N])
ax[1].set_yticks(range(5)[::-1]); ax[1].set_yticklabels(N,fontsize=6); ax[1].set_xlabel('% ranked 1st',fontsize=6); ax[1].set_title('(b) Sensitivity',fontsize=7)
plt.tight_layout(pad=0.2); plt.savefig('figs/fig4.png',dpi=300); plt.close()
