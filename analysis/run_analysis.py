"""Prespecified analysis v3. Test evaluation happens only after all development fits.
Dependencies are isolated in ../tmp/analysis_deps; no raw files are changed.
"""
import sys,os,json,hashlib,time,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tmp'/'analysis_deps'))
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,lasso_path
from sklearn.model_selection import KFold,GroupKFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
threadpool_limits(1)
DATA=ROOT.parent/'datasets'  # place rollingsales_brooklyn.xlsx and rollingsales_queens.xlsx here
CFG=json.loads((ROOT/'protocol_frozen.json').read_text())
NUM=['log_gross','log_land','age','units','month']
GRIDS={'Ridge':CFG['ridge_alpha_SSE'],'Lasso':CFG['lasso_alpha_meanSSE']}
issues=[]

def load():
 parts=[];hashes={}
 for b in ['Brooklyn','Queens']:
  f=DATA/f'rollingsales_{b.lower()}.xlsx';hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
  d=pd.read_excel(f,header=4);d.columns=[str(c).strip().replace('\n',' ') for c in d]
  d['BOROUGH']=b;parts.append(d)
 d=pd.concat(parts,ignore_index=True);raw=len(d)
 d=d[d['BUILDING CLASS CATEGORY'].astype(str).str.strip().str.match(r'^0[123] ')].copy();family=len(d)
 for c in ['SALE PRICE','GROSS SQUARE FEET','LAND SQUARE FEET','YEAR BUILT','TOTAL UNITS','BLOCK','LOT']:d[c]=pd.to_numeric(d[c],errors='coerce')
 missing=int((d['GROSS SQUARE FEET'].isna()|d['GROSS SQUARE FEET'].le(0)).sum());steps={}
 for c in ['SALE PRICE','GROSS SQUARE FEET','YEAR BUILT','LAND SQUARE FEET']:
  keep=d[c].ge(200000) if c=='SALE PRICE' else d[c].gt(0)
  steps[c]=int((~keep).sum());d=d[keep].copy()
 d['SALE DATE']=pd.to_datetime(d['SALE DATE'])
 for c in ['NEIGHBORHOOD','BUILDING CLASS CATEGORY']:d[c]=d[c].astype(str).str.strip()
 d['property']=d['BOROUGH']+'_'+d['BLOCK'].astype(str)+'_'+d['LOT'].astype(str)
 d['log_gross']=np.log(d['GROSS SQUARE FEET']);d['log_land']=np.log(d['LAND SQUARE FEET'])
 d['age']=2026-d['YEAR BUILT'];d['units']=d['TOTAL UNITS'];d['month']=d['SALE DATE'].dt.month
 d['y']=np.log(d['SALE PRICE'])
 assert d[NUM+['y']].notna().all().all()
 tr=d[d['SALE DATE']<'2026-07-01'].reset_index(drop=True)
 te=d[(d['SALE DATE']>='2026-07-01')&(d['SALE DATE']<'2026-09-01')].reset_index(drop=True)
 audit={'raw':raw,'families':family,'missing_gross_before_price':missing,'sequential_drops':steps,'clean':len(d),'development':len(tr),'test':len(te),'development_unique_properties':tr.property.nunique(),'test_unique_properties':te.property.nunique(),'test_rows_property_seen_in_development':int(te.property.isin(tr.property).sum()),'development_repeated_property_rows':int(tr.property.duplicated().sum()),'source_sha256':hashes}
 (ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2))
 return tr,te

class Encoder:
 def __init__(self,kind):self.kind=kind
 def raw(self,d):
  arr=[d[NUM].to_numpy(float)]
  b=np.column_stack([(d['BUILDING CLASS CATEGORY']==v).to_numpy(float) for v in self.bc[1:]])
  arr.append(b)
  if self.kind!='struct':
   n=np.column_stack([(d.NEIGHBORHOOD==v).to_numpy(float) for v in self.nb[1:]])
   arr.append(n)
   if self.kind=='interaction':arr.append((n[:,:,None]*b[:,None,:]).reshape(len(d),-1))
  return np.column_stack(arr)
 def fit(self,d):
  self.bc=sorted(d['BUILDING CLASS CATEGORY'].unique());self.nb=sorted(d.NEIGHBORHOOD.unique())
  x=self.raw(d);self.keep=x.std(0)>1e-12;x=x[:,self.keep]
  self.mu=x.mean(0);self.sd=x.std(0);return self
 def transform(self,d):return (self.raw(d)[:,self.keep]-self.mu)/self.sd

def model(method,a=None):
 if method=='OLS':return LinearRegression()
 if method=='Ridge':return Ridge(alpha=a,solver='cholesky')
 return Lasso(alpha=a,max_iter=30000,tol=1e-6,selection='cyclic',precompute=True)

def fitpredict(tr,va,kind,method,a=None):
 e=Encoder(kind).fit(tr);x=e.transform(tr);z=e.transform(va)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  m=model(method,a).fit(x,tr.y)
 for w in caught:
  if issubclass(w.category,ConvergenceWarning):issues.append({'kind':kind,'method':method,'alpha':a,'n':len(tr),'warning':str(w.message)})
 return m.predict(z),e,m

def tune(d,kind,method,seed=42,grouped=False):
 grid=GRIDS[method];err=np.zeros(len(grid))
 splitter=GroupKFold(5,shuffle=True,random_state=seed) if grouped else KFold(5,shuffle=True,random_state=seed)
 for ti,vi in splitter.split(d,groups=d.property if grouped else None):
  tr,va=d.iloc[ti],d.iloc[vi];e=Encoder(kind).fit(tr);x=e.transform(tr);z=e.transform(va)
  if method=='Lasso':
   xm=x.mean(0);ym=tr.y.mean();xc=np.asfortranarray(x-xm)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning)
    aa,coef,gaps=lasso_path(xc,tr.y.to_numpy()-ym,alphas=sorted(grid,reverse=True),precompute=xc.T@xc,max_iter=30000,tol=1e-6)
   for w in caught:
    if issubclass(w.category,ConvergenceWarning):issues.append({'kind':kind,'method':method,'n':len(tr),'stage':'CV path','warning':str(w.message)})
   preds=ym+(z-xm)@coef
   for j,a in enumerate(grid):err[j]+=np.sum((va.y.to_numpy()-preds[:,np.argmin(np.abs(aa-a))])**2)
  else:
   # All ridge candidates share one eigendecomposition in this fit fold.
   xm=x.mean(0);ym=tr.y.mean();xc=x-xm
   eig,vec=np.linalg.eigh(xc.T@xc);rhs=vec.T@(xc.T@(tr.y.to_numpy()-ym))
   preds=ym+((z-xm)@vec)@(rhs[:,None]/(np.maximum(eig,0)[:,None]+np.asarray(grid)))
   err+=((va.y.to_numpy()[:,None]-preds)**2).sum(0)
 return float(grid[np.argmin(err)]),float(np.sqrt(err.min()/len(d)))

def rmse(y,p):return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))

def main():
 start=time.time();tr,te=load();names=CFG['models'];oof={k:np.zeros(len(tr)) for k in names};foldrows=[];fits={}
 for fold,(ti,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(tr),1):
  a,b=tr.iloc[ti],tr.iloc[vi]
  for name in names:
   print('Fitting outer',fold,name,flush=True)
   penalty=None
   if name=='mean':pred=np.full(len(b),a.y.mean())
   elif name=='area':pred=LinearRegression().fit(a[['GROSS SQUARE FEET']],a.y).predict(b[['GROSS SQUARE FEET']])
   else:
    kind,method=name.split('_');penalty=tune(a,kind,method,42+fold)[0] if method!='OLS' else None
    pred,_,_=fitpredict(a,b,kind,method,penalty)
   oof[name][vi]=pred;foldrows.append({'fold':fold,'model':name,'rmse':rmse(b.y,pred),'alpha':penalty})
  print('Outer fold',fold,'done, seconds',round(time.time()-start),flush=True)
 pd.DataFrame(foldrows).to_csv(ROOT/'development_folds.csv',index=False)
 # Freeze full-development fits and all learning-curve settings before test scoring.
 for name in names:
  if '_' not in name:continue
  kind,method=name.split('_');penalty=tune(tr,kind,method)[0] if method!='OLS' else None
  e=Encoder(kind).fit(tr);x=e.transform(tr);m=model(method,penalty).fit(x,tr.y)
  fits[name]=(e,m);print('Full fit',name,'p',x.shape[1],'alpha',penalty,flush=True)
  # Input matrix rank and singular-value diagnostics on development only.
  s=np.linalg.svd(x,compute_uv=False);rank=int(np.sum(s>np.finfo(float).eps*max(x.shape)*s[0]))
  fits[name]+=(penalty,x.shape[1],rank)
 learn=[];lc_models=[]
 for n in CFG['learning_sizes']:
  for seed in ([42] if n==len(tr) else CFG['learning_seeds']):
   ix=np.random.default_rng(seed).choice(len(tr),n,replace=False);sample=tr.iloc[ix]
   for kind in ['add','interaction']:
    penalty=tune(sample,kind,'Ridge',seed)[0]
    e=Encoder(kind).fit(sample);x=e.transform(sample)
    for method in ['OLS','Ridge']:
     m=model(method,penalty if method=='Ridge' else None).fit(x,sample.y)
     lc_models.append((n,seed,kind,method,e,m,penalty if method=='Ridge' else None))
   print('Learning fits n',n,'seed',seed,'seconds',round(time.time()-start),flush=True)
 # Grouped development-only robustness check: property never crosses fit/validation.
 group_pred={k:np.zeros(len(tr)) for k in ['add_OLS','add_Ridge']}
 for fold,(ti,vi) in enumerate(GroupKFold(5,shuffle=True,random_state=42).split(tr,groups=tr.property),1):
  a,b=tr.iloc[ti],tr.iloc[vi]
  for name in group_pred:
   kind,method=name.split('_');penalty=tune(a,kind,method,42+fold,True)[0] if method=='Ridge' else None
   group_pred[name][vi]=fitpredict(a,b,kind,method,penalty)[0]
 pd.DataFrame({'property':tr.property,'y':tr.y,**oof}).to_csv(ROOT/'development_predictions.csv',index=False)
 (ROOT/'pretest_fits.json').write_text(json.dumps({k:{'alpha':v[2],'columns':v[3],'rank':v[4]} for k,v in fits.items()},indent=2))
 print('ALL DEVELOPMENT CHOICES FROZEN. Beginning prespecified temporal evaluation.',flush=True)
 testpred={'mean':np.full(len(te),tr.y.mean()),'area':LinearRegression().fit(tr[['GROSS SQUARE FEET']],tr.y).predict(te[['GROSS SQUARE FEET']])}
 for name,(e,m,*_) in fits.items():testpred[name]=m.predict(e.transform(te))
 pd.DataFrame({'property':te.property,'date':te['SALE DATE'],'y':te.y,**testpred}).to_csv(ROOT/'temporal_predictions.csv',index=False)
 results=[];bootrow={k:[] for k in names};bootgroup={k:[] for k in names}
 sq={k:(te.y.to_numpy()-testpred[k])**2 for k in names}
 codes,props=pd.factorize(te.property);cnt=np.bincount(codes)
 sums={k:np.bincount(codes,weights=sq[k]) for k in names}
 rng=np.random.default_rng(42)
 for _ in range(2000):
  ix=rng.integers(len(te),size=len(te));g=rng.integers(len(props),size=len(props));den=cnt[g].sum()
  for k in names:
   bootrow[k].append(float(np.sqrt(sq[k][ix].mean())))
   bootgroup[k].append(float(np.sqrt(sums[k][g].sum()/den)))
 for k in names:
  fr=[r['rmse'] for r in foldrows if r['model']==k]
  results.append({'model':k,'cv_pooled':rmse(tr.y,oof[k]),'cv_fold_mean':np.mean(fr),'cv_fold_sd':np.std(fr,ddof=1),'test_rmse':rmse(te.y,testpred[k]),'test_mae_log':float(np.mean(np.abs(te.y-testpred[k]))),'row_low':np.quantile(bootrow[k],.025),'row_high':np.quantile(bootrow[k],.975),'cluster_low':np.quantile(bootgroup[k],.025),'cluster_high':np.quantile(bootgroup[k],.975),'alpha':fits[k][2] if k in fits else None,'columns':fits[k][3] if k in fits else None})
 pd.DataFrame(results).to_csv(ROOT/'model_results.csv',index=False)
 pairs=[('add_OLS','struct_OLS'),('add_Ridge','add_OLS'),('add_Lasso','add_OLS'),('interaction_OLS','add_OLS'),('interaction_Ridge','interaction_OLS'),('interaction_Lasso','interaction_OLS'),('interaction_Ridge','add_OLS')]
 pairsout=[]
 for a,b in pairs:
  diff=np.array(bootrow[a])-bootrow[b];cl=np.array(bootgroup[a])-bootgroup[b]
  pairsout.append({'model':a,'reference':b,'delta_rmse':rmse(te.y,testpred[a])-rmse(te.y,testpred[b]),'row_low':np.quantile(diff,.025),'row_high':np.quantile(diff,.975),'cluster_low':np.quantile(cl,.025),'cluster_high':np.quantile(cl,.975)})
 pd.DataFrame(pairsout).to_csv(ROOT/'paired_comparisons.csv',index=False)
 lc_pred={}
 for n,seed,kind,method,e,m,alpha in lc_models:
  pred=m.predict(e.transform(te));lc_pred[f'{n}_{seed}_{kind}_{method}']=pred
  learn.append({'n':n,'seed':seed,'features':kind,'method':method,'alpha':alpha,'test_rmse':rmse(te.y,pred)})
 pd.DataFrame(learn).to_csv(ROOT/'learning_runs.csv',index=False)
 np.savez_compressed(ROOT/'learning_predictions.npz',**lc_pred)
 summary={'grouped_cv':{k:rmse(tr.y,p) for k,p in group_pred.items()},'unseen_test_neighborhood_rows':int((~te.NEIGHBORHOOD.isin(tr.NEIGHBORHOOD)).sum()),'convergence_warnings':issues,'seconds':time.time()-start}
 (ROOT/'diagnostics.json').write_text(json.dumps(summary,indent=2))
 print(pd.DataFrame(results).to_string(index=False),flush=True);print(pd.DataFrame(pairsout).to_string(index=False),flush=True)
 plot(pd.DataFrame(results),pd.DataFrame(learn),lc_pred,te.y.to_numpy())

def plot(results,learn,preds,y):
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(10,5))
 ax.bar(results.model,results.test_rmse,color=['#bdc9d7','#9aadc1']+['#3272ae']*4+['#173f67']*3)
 ax.errorbar(range(len(results)),results.test_rmse,yerr=[results.test_rmse-results.cluster_low,results.cluster_high-results.test_rmse],fmt='none',color='black',capsize=3)
 ax.tick_params(axis='x',rotation=40);ax.set(ylabel='Temporal-test RMSE of natural-log price',title='July-August 2026: all prespecified models')
 ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(ROOT/'model_comparison.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
 for ax,kind in zip(axs,['add','interaction']):
  for method,color in [('OLS','#3272ae'),('Ridge','#d77832')]:
   means=[];low=[];high=[]
   for n in CFG['learning_sizes']:
    rows=learn[(learn.n==n)&(learn.features==kind)&(learn.method==method)]
    arr=np.array([preds[f'{n}_{int(s)}_{kind}_{method}'] for s in rows.seed])
    # Mean RMSE over matched training runs; evaluation-row resampling conditional on fits.
    rng=np.random.default_rng(42);bs=[]
    for _ in range(1000):
     ix=rng.integers(len(y),size=len(y));bs.append(np.sqrt(((arr[:,ix]-y[ix])**2).mean(1)).mean())
    means.append(rows.test_rmse.mean());lo,hi=np.quantile(bs,[.025,.975]);low.append(lo);high.append(hi)
   ax.plot(CFG['learning_sizes'],means,'o-',color=color,label=method)
   ax.fill_between(CFG['learning_sizes'],low,high,color=color,alpha=.14)
  ax.set(title=kind.title()+' features',xlabel='Training sales',xscale='log');ax.grid(alpha=.15);ax.legend()
 axs[0].set_ylabel('Mean temporal-test RMSE of log price')
 fig.suptitle('Learning curves: 5 fixed subset seeds (full sample fitted once)')
 fig.text(.07,.01,'Bands: 95% row-bootstrap intervals conditional on fitted models; training-run variation is reported separately.',fontsize=8)
 fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(ROOT/'learning_curves.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
