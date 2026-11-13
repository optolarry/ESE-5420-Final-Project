from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent
DEPS=ROOT.parent/'tmp'/'analysis_deps'
if not DEPS.is_dir():
    DEPS=Path(r'C:\Users\zxg\Documents\ChatGPT\ESE 5420 Final Project\tmp\analysis_deps')
sys.path.insert(0,str(DEPS))
import numpy as np,pandas as pd,sklearn,scipy
r=pd.read_csv(ROOT/'model_summary.csv');pairs=pd.read_csv(ROOT/'paired_vs_OLS.csv');fold=pd.read_csv(ROOT/'fold_results.csv');pred=pd.read_csv(ROOT/'development_oof_predictions.csv')
assert abs(float(r.loc[r.model=='OLS','oof_rmse'].iloc[0])-0.34325242231765835)<1e-9
assert len(pred)==10737 and pd.to_datetime(pred.date).max()<pd.Timestamp('2026-07-01')
for row in r.itertuples():
 mse=float(np.mean((pred.y-pred[row.model])**2))
 assert abs(mse-row.oof_mse)<1e-12
 assert abs(np.sqrt(mse)-row.oof_rmse)<1e-12
assert len(fold)==20 and fold.groupby('model').size().eq(5).all()
labels={'OLS':'加性 OLS','HGB':'梯度提升树 HGB','RF':'随机森林','SplineRidge':'样条回归 + Ridge'}
text='# 新方法比较：只使用开发数据\n\n'
text+='本轮未打开新下载的 2026 年 9 月数据，没有给新模型计算 7—8 月测试成绩。全部数字来自原始开发数据的外层交叉验证。\n\n'
text+='| 方法 | log-price MSE | RMSE | 相对 OLS 的 MSE 降幅 |\n|---|---:|---:|---:|\n'
for row in r.itertuples():text+=f'| {labels[row.model]} | {row.oof_mse:.6f} | {row.oof_rmse:.6f} | {row.MSE_reduction_vs_OLS_pct:.2f}% |\n'
text+='\n## 相对 OLS 的配对 RMSE 差值\n\n负数代表更低误差。区间仅条件于已拟合的折外预测，不完整包含模型训练及折之间的依赖。\n\n'
for row in pairs.itertuples():text+=f'- {labels[row.model]}: {row.delta_rmse:+.6f}; 95% 条件区间 [{row.conditional_low:+.6f}, {row.conditional_high:+.6f}]。\n'
text+='\nHGB 和随机森林学习非线性与交互关系；样条回归检验平滑非线性是否足够。不同树方法不会自动胜过线性模型，结果应完整保留。\n\n'
text+='调参仅使用每个外层训练折内的三折验证。外层五折、seed=42，与原有 OLS 评估一致。类别编码和变换也在内层训练数据内拟合。\n\n'
text+='下一步若选择其中一种方法，应先确定超参数搜索和最终训练程序，然后才打开独立 9 月样本。不能反复查看 9 月成绩选择方法。若以后查看 7—8 月成绩，应标为已有测试集上的后续探索，不能假称从未查看。\n'
(ROOT/'results_zh.md').write_text(text,encoding='utf-8')
(ROOT/'environment.json').write_text(json.dumps({'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pd.__version__},indent=2))
print(text)
print('Checks: 10,737 development rows, 5 outer scores/model, MSE/RMSE match saved predictions.')
