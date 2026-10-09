"""Build result tables and supplementary figures from saved QoI-FDM evidence."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.qoi_fdm import NAMES
from utils.cst_reduction import cst_surfaces
D=ROOT/'data/qoi_fdm'; F=ROOT/'figures/qoi_fdm'
m=pd.read_csv(D/'metrics.csv'); s=m.groupby('qoi').mean(numeric_only=True).reindex(NAMES)
o=pd.read_csv(D/'optimization.csv'); inv=pd.read_csv(D/'inverse.csv'); meta=json.loads((D/'metadata.json').read_text())
selected=[]
for n in NAMES:
    g=o[(o.qoi==n)&o.success&o.geometry_valid&(o.confidence>=.75)]
    selected.append(g.loc[g.q_actual.idxmin()])
best=pd.DataFrame(selected).set_index('qoi').reindex(NAMES); best.to_csv(D/'selected_candidates.csv')
escape=lambda n:n.replace('_',r'\_')
lines=[r'\subsection{Cohort and conditional preservation}',f"Of {meta['n_raw']} original shapes, {meta['n_confident']} pass the confidence screen; {meta['n_common']} also reach every prescribed lift. The additional 44 exclusions change the population from the preceding CST study. Each split has 360 training, 83 validation, and 111 test shapes. There are 432 kernel fits, 360 decoded holdout evaluations, and 96 optimized candidates.",r'\begin{center}\small\begin{tabular}{lrrrrr}\toprule Objective & NRMSE & Field error & Recall & Hull & Shape RMS\\\midrule']
for n,r in s.iterrows():
    lines.append(f"{escape(n)} & {r.nrmse:.3f} & {r.field_error:.3f} & {100*r.recall5:.1f}\\% & {100*r.hull_coverage:.1f}\\% & {100*r.shape_inverse_rmse:.3f}\\% \\")
lines += [r'\bottomrule\end{tabular}\end{center}',r'Table entries are means across three splits; shape RMS is percent chord and hull is supported test fraction. Spearman correlations average above 0.9997 for every objective. Nevertheless, eleven of twelve mean field errors exceed one. Lift tracking is the only objective with a mean error below the training-mean baseline (0.918). Five-neighbor recall is only 3.2--5.8\%. The ten-neighbor shape errors are 0.927--1.028\% chord, compared with 0.841\% for the mean shape on the same full test sets. Barycentric inverse errors are 0.897--1.040\% on supported subsets; those subsets differ from the full-test baseline, so that comparison is descriptive rather than a paired baseline test.',r'Best drag/lift has the largest average scalar error (0.116), with split values 0.058, 0.102, and 0.189 despite almost perfect ranks. Worst drag ranges from 0.007 to 0.225. Thus reliable ordering need not imply accurate scalar distances or stable tail reconstruction. The selected quantity weight is 16 in 34 of 36 fits, with the remaining two selecting 4. This boundary preference suggests that scalar-only validation rewards progressively dominant target information; it does not establish that the search has found an optimal field/shape compromise.',r'\subsection{Decoded holdout queries}',r'\begin{center}\small\begin{tabular}{lrrr}\toprule Objective & Decoded QoI NRMSE & Median round trip & Minimum confidence\\\midrule']
for n in NAMES:
    g=inv[inv.qoi==n]
    lines.append(f"{escape(n)} & {np.sqrt(np.mean(g.q_error_standardized**2)):.3f} & {g['roundtrip'].median():.3f} & {g.confidence.min():.3f} \\")
lines += [r'\bottomrule\end{tabular}\end{center}',r'These 30 queries per objective are not the full holdout set. Their decoded QoI errors are substantially larger than conditional scalar interpolation errors. Multipoint drag reaches approximately one training standard deviation of error and a median round trip of 0.659 training latent distances. Even the better round trips remain nonzero. All these decoded evaluations exceed the confidence threshold, so model-confidence screening alone does not detect representation inconsistency.',r'\subsection{Optimized candidate re-evaluation}',r'\begin{center}\small\begin{tabular}{lrrrr}\toprule Objective & Coarse gain (\%) & Fine gain (\%) & Larger model (\%) & Round trip\\\midrule']
for n,r in best.iterrows():
    lines.append(f"{escape(n)} & {r.gain_percent:.4f} & {r.fine_gain_percent:.4f} & {r.variant_gain_percent:.4f} & {r['roundtrip']:.3f} \\")
lines += [r'\bottomrule\end{tabular}\end{center}',r'All 96 optimizations report success and pass the coarse-grid confidence and geometry screens. This does not imply improvement. Mean efficiency is the only objective with a selected positive gain above 0.001\%: 0.680\% on the original grid, 0.678\% on the refined grid, and 1.160\% with the larger model. Its round-trip error is 0.455, demonstrating useful task improvement despite poor coordinate consistency in this one case. This is a modest model-based result, not a general validation of the inverse.',r'Most other selected candidates are effectively the incumbent: the decoder uses a tiny interior buffer at the hull boundary, so changes of order $10^{-4}$ percent should not be treated as design advances. Minimum drag changes sign with network size. Worst prescribed-lift drag deteriorates by 1.183\% on the original grid and 0.514\% with the larger model; its tiny round trip does not imply a good objective value, because it can return a different training neighborhood. The original incumbent must be retained whenever a candidate is worse. Neither a feasible hull point nor optimizer success is sufficient for adoption.',r'\subsection{Interpretation across objectives}',r'The objective-correlation matrix identifies shared and competing rankings without comparing arbitrary diffusion-axis signs. The embedding panels show strongly curved or thin scalar orderings. Together with high target weights, near-perfect scalar ranks, poor field recall, and inaccurate geometric inversion, the evidence supports interpreting these maps primarily as objective-organized descriptors. It does not support treating the two coordinates as a generally invertible shape parameterization. The mean-efficiency candidate is a useful exception for further validation, while the prescribed-lift results provide no meaningful optimization gain in this run.']
# TeX row endings need two literal backslashes.
lines=[line+'\\' if line.endswith(' \\') else line for line in lines]
(ROOT/'report/qoi_fdm_results.tex').write_text('\n\n'.join(lines))
plt.style.use(ROOT/'utils/academic.mplstyle')
fig,axes=plt.subplots(1,2,figsize=(10,4))
positions=np.arange(len(NAMES))
for col,mark,label in [('gain_percent','o','Original grid'),('fine_gain_percent','s','Refined grid'),('variant_gain_percent','^','Larger model')]:
    axes[0].plot(best[col],positions,mark,ms=4,label=label)
axes[0].set_yticks(positions,[n.replace('_',' ') for n in NAMES]); axes[0].axvline(0,color='grey',lw=.6); axes[0].set_xlabel('Selected candidate gain (%)'); axes[0].legend(fontsize=7)
arc=np.load(D/'embeddings.npz'); cp=arc['mean_efficiency_candidate_parameters']; j=int(best.loc['mean_efficiency','start']); x=np.linspace(0,1,301)
for p,label,style in [(cp[0],'Incumbent','--'),(cp[j],'Efficiency candidate','-')]:
    u,l=cst_surfaces(p,x); axes[1].plot(x,u[0],style,label=label); axes[1].plot(x,l[0],style,color=axes[1].lines[-1].get_color())
axes[1].set_aspect('equal'); axes[1].set_xlabel('$x/c$'); axes[1].set_ylabel('$y/c$'); axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F/'optimization.pdf'); plt.close(fig)
