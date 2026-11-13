"""Development-only nonlinear comparison. Never opens new monthly files or test predictions.
Only the original houses_clean.csv is read; July-August rows are excluded before fitting.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEPS=ROOT.parent/'tmp'/'analysis_deps'
if not DEPS.is_dir():
    DEPS=Path(r'C:\Users\zxg\Documents\ChatGPT\ESE 5420 Final Project\tmp\analysis_deps')
sys.path.insert(0,str(DEPS))
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler,SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,GridSearchCV
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
threadpool_limits(1)
NUM=['log_gross','log_land','age','units','month'];CAT=['NEIGHBORHOOD','BUILDING CLASS CATEGORY']
SOURCE=ROOT.parent/'datasets'/'houses_clean.csv'

def pipeline(name):
 if name=='HGB':
  enc=ColumnTransformer([('num','passthrough',NUM),('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan),CAT)],sparse_threshold=0)
  return Pipeline([('encode',enc),('model',HistGradientBoostingRegressor(categorical_features=[False]*5+[True]*2,learning_rate=.05,max_iter=250,early_stopping=False,random_state=42))])
 num=SplineTransformer(n_knots=4,degree=3,include_bias=False) if name=='SplineRidge' else 'passthrough'
 enc=ColumnTransformer([('num',num,NUM),('cat',OneHotEncoder(handle_unknown='ignore',drop='first' if name in ['OLS','SplineRidge'] else None,sparse_output=False),CAT)],sparse_threshold=0)
 if name=='RF':return Pipeline([('encode',enc),('model',RandomForestRegressor(n_estimators=200,n_jobs=4,random_state=42))])
 return Pipeline([('encode',enc),('scale',StandardScaler()),('model',LinearRegression() if name=='OLS' else Ridge(solver='cholesky'))])

GRIDS={'HGB':{'model__max_leaf_nodes':[15,31],'model__min_samples_leaf':[10,30],'model__l2_regularization':[0.,10.]},'RF':{'model__max_features':[.7,1.],'model__min_samples_leaf':[2,10,30]},'SplineRidge':{'encode__num__n_knots':[4,6],'model__alpha':[.01,1.,100.,10000.]}}

def main():
 start=time.time();d=pd.read_csv(SOURCE,parse_dates=['SALE DATE'])
 # No target statistics or predictions for excluded months are calculated.
 d=d[d['SALE DATE']<'2026-07-01'].reset_index(drop=True)
 assert len(d)==10737 and d['SALE DATE'].max()<pd.Timestamp('2026-07-01')
 d['log_gross']=np.log(d['GROSS SQUARE FEET']);d['log_land']=np.log(d['LAND SQUARE FEET']);d['age']=2026-d['YEAR BUILT'];d['units']=d['TOTAL UNITS'];d['month']=d['SALE DATE'].dt.month
 X=d[NUM+CAT];y=np.log(d['SALE PRICE'].to_numpy())
 cfg={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'development_n':len(d),'last_date':str(d['SALE DATE'].max()),'outer_folds':5,'outer_seed':42,'inner_folds':3,'inner_seeds':[43,44,45,46,47],'grids':GRIDS,'score':'log-price MSE for selection; MSE and RMSE reported','RF_trees':200,'HGB_iterations':250,'HGB_learning_rate':.05,'September_accessed':False,'July_August_scored':False,'bootstrap':'2000 paired row resamples of fixed OOF errors, conditional on fits; folds and training uncertainty not fully covered'}
 (ROOT/'protocol.json').write_text(json.dumps(cfg,indent=2))
 names=['OLS','HGB','RF','SplineRidge'];pred={k:np.zeros(len(d)) for k in names};rows=[]
 for fold,(ti,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(X),1):
  for name in names:
   print('Starting',fold,name,'seconds',round(time.time()-start),flush=True)
   if name=='OLS':m=pipeline(name).fit(X.iloc[ti],y[ti]);pars={};inner=None
   else:
    search=GridSearchCV(pipeline(name),GRIDS[name],scoring='neg_mean_squared_error',cv=KFold(3,shuffle=True,random_state=42+fold),n_jobs=1,refit=True,error_score='raise')
    search.fit(X.iloc[ti],y[ti]);m=search.best_estimator_;pars=search.best_params_;inner=float(-search.best_score_)
    pd.DataFrame(search.cv_results_).to_csv(ROOT/f'tuning_{name}_fold{fold}.csv',index=False)
   p=m.predict(X.iloc[vi]);pred[name][vi]=p;mse=float(np.mean((y[vi]-p)**2))
   rows.append({'fold':fold,'model':name,'mse':mse,'rmse':np.sqrt(mse),'params':json.dumps(pars),'inner_best_mse':inner})
   pd.DataFrame(rows).to_csv(ROOT/'fold_results.csv',index=False)
   print('Done',fold,name,'RMSE',round(np.sqrt(mse),6),'params',pars,flush=True)
 pd.DataFrame({'date':d['SALE DATE'],'y':y,**pred}).to_csv(ROOT/'development_oof_predictions.csv',index=False)
 err={k:(y-v)**2 for k,v in pred.items()};bs={k:np.zeros(2000) for k in names};rng=np.random.default_rng(42)
 for b in range(2000):
  ix=rng.integers(len(y),size=len(y))
  for k in names:bs[k][b]=err[k][ix].mean()
 out=[];paired=[]
 for k in names:
  scores=[r['rmse'] for r in rows if r['model']==k];lo,hi=np.quantile(np.sqrt(bs[k]),[.025,.975])
  out.append({'model':k,'oof_mse':err[k].mean(),'oof_rmse':np.sqrt(err[k].mean()),'fold_rmse_sd':np.std(scores,ddof=1),'conditional_rmse_low':lo,'conditional_rmse_high':hi,'MSE_reduction_vs_OLS_pct':100*(1-err[k].mean()/err['OLS'].mean())})
  if k!='OLS':
   diff=np.sqrt(bs[k])-np.sqrt(bs['OLS']);lo,hi=np.quantile(diff,[.025,.975])
   paired.append({'model':k,'reference':'OLS','delta_rmse':np.sqrt(err[k].mean())-np.sqrt(err['OLS'].mean()),'conditional_low':lo,'conditional_high':hi})
 pd.DataFrame(out).to_csv(ROOT/'model_summary.csv',index=False);pd.DataFrame(paired).to_csv(ROOT/'paired_vs_OLS.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4.5));vals=np.array([r['oof_rmse'] for r in out]);low=np.array([r['conditional_rmse_low'] for r in out]);high=np.array([r['conditional_rmse_high'] for r in out])
 ax.bar(names,vals,color=['#aab7c8','#28639b','#488f83','#d28740']);ax.errorbar(range(4),vals,yerr=[vals-low,high-vals],fmt='none',color='black',capsize=5)
 for i,v in enumerate(vals):ax.text(i,high[i]+.006,f'{v:.4f}',ha='center')
 ax.set(ylabel='Nested-CV RMSE of log sale price',title='Development data only | September untouched',ylim=(0,max(high)*1.18));ax.spines[['top','right']].set_visible(False)
 fig.text(.08,.015,'Bars: conditional 95% bootstrap intervals from fixed OOF errors; no July-August scoring.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig)
 print(pd.DataFrame(out).to_string(index=False),flush=True);print(pd.DataFrame(paired).to_string(index=False),flush=True);print('Total seconds',time.time()-start,flush=True)

if __name__=='__main__':main()
