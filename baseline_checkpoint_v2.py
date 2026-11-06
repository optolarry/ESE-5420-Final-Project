"""Run with Python and numpy, pandas, openpyxl, matplotlib installed.
Inputs are ../datasets/rollingsales_{brooklyn,queens}.xlsx. Final test is never scored.
Ridge objective: SSE + alpha*sum(beta**2); unpenalized intercept.
"""
from pathlib import Path
import sys
# Use the already installed Codex copy only if openpyxl is missing locally.
try:
    import openpyxl
except ModuleNotFoundError:
    bundled = Path(r"C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1030-493-0_26.1030.493.0_x64__3k8sg7r9htsxt\dependencies\python\Lib\site-packages")
    if (bundled / "openpyxl").is_dir():
        sys.path.append(str(bundled))
    else:
        raise ImportError("Install openpyxl with: python -m pip install openpyxl")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GRID = np.logspace(-2, 4, 13)

def folds(n, seed):
    for val in np.array_split(np.random.RandomState(seed).permutation(n), 5):
        mask = np.ones(n, bool)
        mask[val] = False
        yield np.flatnonzero(mask), np.sort(val)

def features(tr, va):
    cats = {c: sorted(tr[c].unique()) for c in ['NEIGHBORHOOD','BUILDING CLASS CATEGORY']}
    def transform(d):
        parts = [np.column_stack([np.log(d['GROSS SQUARE FEET']),np.log(d['LAND SQUARE FEET']),2026-d['YEAR BUILT'],d['TOTAL UNITS'],d['SALE DATE'].dt.month])]
        for c, levels in cats.items():
            parts.append(np.column_stack([(d[c]==v).to_numpy(float) for v in levels[1:]]))
        return np.column_stack(parts).astype(float)
    x,z = transform(tr),transform(va)
    mu,sd = x.mean(0),x.std(0)
    sd[sd==0]=1
    return (x-mu)/sd,(z-mu)/sd

def predictions(x,y,z,alphas):
    xm,ym=x.mean(0),y.mean()
    xc=x-xm
    eig,vec=np.linalg.eigh(xc.T@xc)
    rhs=vec.T@(xc.T@(y-ym))
    return ym+((z-xm)@vec)@(rhs[:,None]/(np.maximum(eig,0)[:,None]+np.asarray(alphas)))

def ols(x,y,z):
    xm,ym=x.mean(0),y.mean()
    return ym+(z-xm)@np.linalg.lstsq(x-xm,y-ym,rcond=None)[0]

def tune(d,seed):
    sums=np.zeros(len(GRID))
    for ti,vi in folds(len(d),seed):
        tr,va=d.iloc[ti],d.iloc[vi]
        x,z=features(tr,va)
        pred=predictions(x,tr.LOG_PRICE.to_numpy(),z,GRID)
        sums+=((va.LOG_PRICE.to_numpy()[:,None]-pred)**2).sum(0)
    return float(GRID[np.argmin(sums)])

def main():
    parts=[]
    for b in ['Brooklyn','Queens']:
        r=pd.read_excel(HERE.parent/'datasets'/f'rollingsales_{b.lower()}.xlsx',header=4)
        r.columns=[str(c).strip().replace('\n',' ') for c in r]
        r['BOROUGH']=b
        parts.append(r)
    raw=pd.concat(parts,ignore_index=True)
    d=raw[raw['BUILDING CLASS CATEGORY'].astype(str).str.strip().str.match(r'^0[123] ')].copy()
    for c in ['SALE PRICE','GROSS SQUARE FEET','LAND SQUARE FEET','YEAR BUILT','TOTAL UNITS']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    steps=[]
    for c in ['SALE PRICE','GROSS SQUARE FEET','YEAR BUILT','LAND SQUARE FEET']:
        keep=d[c].ge(200000) if c=='SALE PRICE' else d[c].gt(0)
        steps.append(int((~keep).sum()))
        d=d[keep].copy()
    d['SALE DATE']=pd.to_datetime(d['SALE DATE'],errors='raise')
    for c in ['NEIGHBORHOOD','BUILDING CLASS CATEGORY']:d[c]=d[c].astype(str).str.strip()
    d['LOG_PRICE']=np.log(d['SALE PRICE'])
    assert np.isfinite(d[['LOG_PRICE','TOTAL UNITS']].to_numpy()).all()
    train=d[d['SALE DATE']<'2026-07-01'].reset_index(drop=True)
    ntest=int((d['SALE DATE']>='2026-07-01').sum())
    names=['Mean','GrossSF','OLS','Ridge']
    errors={k:np.zeros(len(train)) for k in names}
    scores={k:[] for k in names}
    selected=[]
    for fold,(ti,vi) in enumerate(folds(len(train),42),1):
        tr,va=train.iloc[ti],train.iloc[vi]
        y,yv=tr.LOG_PRICE.to_numpy(),va.LOG_PRICE.to_numpy()
        x,z=features(tr,va)
        alpha=tune(tr,42+fold)
        selected.append(alpha)
        g,h=tr[['GROSS SQUARE FEET']].to_numpy(float),va[['GROSS SQUARE FEET']].to_numpy(float)
        gm,gs=g.mean(0),g.std(0)
        pred={'Mean':np.full(len(va),y.mean()),'GrossSF':ols((g-gm)/gs,y,(h-gm)/gs),'OLS':ols(x,y,z),'Ridge':predictions(x,y,z,[alpha])[:,0]}
        for k in names:
            errors[k][vi]=(yv-pred[k])**2
            scores[k].append(float(np.sqrt(errors[k][vi].mean())))
        print('Fold',fold,'alpha',alpha,{k:round(scores[k][-1],6) for k in names},flush=True)
    alpha=tune(train,42)
    rng=np.random.default_rng(42)
    boots={k:np.empty(2000) for k in names}
    for b in range(2000):
        ix=rng.integers(0,len(train),len(train))
        for k in names:boots[k][b]=np.sqrt(errors[k][ix].mean())
    results={k:(float(np.sqrt(errors[k].mean())),np.quantile(boots[k],[.025,.975]),float(np.mean(scores[k])),float(np.std(scores[k],ddof=1))) for k in names}
    p=features(train,train.iloc[:1])[0].shape[1]
    lines='\n'.join(f'{k}: pooled OOF RMSE {r[0]:.4f}, 95% interval [{r[1][0]:.4f}, {r[1][1]:.4f}]; fold mean {r[2]:.4f}, SD {r[3]:.4f}.' for k,r in results.items())
    txt=f'''# Predicting House Sale Prices in Queens and Brooklyn, NYC

## Group
Solo project; Xiaoguang (instructor-approved).

## Changes since the proposal
Dropped classification; reserved the final two months for testing. Added CV-tuned ridge alongside OLS. Interactions and smaller training sets remain planned.

## Data
NYC DOF Rolling Sales, Brooklyn and Queens, September 2, 2025-August 31, 2026. Of {len(raw):,} raw rows, 26,908 are 1-3 family dwellings; 123 have missing/nonpositive gross area before price filtering. Sequential removals: {steps[0]:,} prices below $200,000; {steps[1]} missing/nonpositive gross areas; {steps[2]} missing/nonpositive construction years; {steps[3]} invalid land areas. Final N={len(d):,}; output=natural-log sale price. Inputs: log gross area, log land area, age, units, month, neighborhood and building-class indicators ({p} encoded columns on all earlier sales). Borough omitted as redundant with neighborhood.

## Split
Earlier sales: N={len(train):,}; July-August 2026 final test: N={ntest:,}, never scored. Shuffled 5-fold outer CV, seed 42, on earlier sales only. Encoding/scaling fitted within training folds. Ridge uses inner 5-fold tuning, seeds 43-47. Random CV measures within-period generalization; temporal testing is reserved for later sales.

## Baseline
OLS with mean and gross-area-only references. Ridge objective: SSE+alpha*sum(beta^2), intercept unpenalized; all ridge inputs standardized within folds. Grid: 13 log-spaced alpha values, 0.01-10,000. Inner-CV choices: {', '.join(f'{a:g}' for a in selected)}. Five-fold tuning on all earlier sales selects alpha={alpha:g} for eventual final fitting; reported ridge performance uses outer CV, separate from tuning.

## Result
{lines}
Error bars: 2,000 paired row-bootstrap resamples of fixed out-of-fold (OOF) squared errors, seed 42; percentile intervals. These measure evaluation-sample variability conditional on fitted folds, not full retraining uncertainty. Fold SD is reported separately.

## Issues found
Unseen neighborhoods use the fit-fold reference category. Repeated-property sales and duplicate-looking records remain; dependence may make row-bootstrap intervals optimistic. Low price is a filtering rule, not proof of a non-market transfer.

## Next steps
Xiaoguang: before November 16, add neighborhood/class interactions, ridge/lasso comparisons and training-size experiments (500-5,000). Develop comparisons on earlier sales; reserve final temporal-test learning curves and paired model-minus-OLS bootstrap for the frozen analysis.

## Code
https://github.com/optolarry/ESE-5420-Final-Project
Local script: baseline_checkpoint_v2.py. Repository upload/access remains to be verified.
'''
    assert len(txt.split())<500,len(txt.split())
    assert [l[3:] for l in txt.splitlines() if l.startswith('## ')]==['Group','Changes since the proposal','Data','Split','Baseline','Result','Issues found','Next steps','Code']
    (HERE/'baseline_v2.txt').write_text(txt,encoding='utf-8')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    fig.suptitle('House sale price prediction | Baseline checkpoint',fontsize=17,fontweight='bold',y=.98)
    fig.text(.5,.91,'Earlier sales only; July-August 2026 final test remains untouched',ha='center',color='#526174')
    vals=np.array([results[k][0] for k in names]);ci=np.array([results[k][1] for k in names])
    axes[0].bar(names,vals,color=['#c4cfdc','#9eafc4','#3272ae','#163e66'],width=.62)
    axes[0].errorbar(range(4),vals,yerr=np.stack([vals-ci[:,0],ci[:,1]-vals]),fmt='none',ecolor='#111827',capsize=5)
    for i,v in enumerate(vals):axes[0].text(i,ci[i,1]+.012,f'{v:.4f}',ha='center',fontweight='bold')
    axes[0].set(title='OOF RMSE with 95% bootstrap intervals',ylabel='RMSE of natural-log sale price',ylim=(0,.65))
    for k,c in zip(names,['#a7b5c7','#7f96b3','#3272ae','#163e66']):axes[1].plot(range(1,6),scores[k],marker='o',label=k,color=c,lw=2,linestyle='--' if k=='Ridge' else '-')
    axes[1].set(title='Validation RMSE in each outer fold',xlabel='Outer fold',ylabel='RMSE of natural-log sale price',xticks=range(1,6))
    axes[1].legend(frameon=False,loc='center right')
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.06,.035,'Ridge: inner 5-fold tuning. Intervals: 2,000 fixed-OOF-error resamples; not retraining uncertainty.',fontsize=10,color='#526174')
    fig.subplots_adjust(top=.79,bottom=.17,left=.07,right=.98,wspace=.3)
    fig.savefig(HERE/'baseline_v2.png',dpi=180,facecolor='white')
    plt.close(fig)
    print('FEATURES',p,'WORDS',len(txt.split()),'FINAL ALPHA',alpha,flush=True)
    for k,r in results.items():print(k,r,flush=True)

if __name__=='__main__':main()
