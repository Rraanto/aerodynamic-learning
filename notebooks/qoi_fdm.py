"""Reproduce the objective comparison: OPENBLAS_NUM_THREADS=1 python notebooks/qoi_fdm.py."""
import sys, json, hashlib, platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from utils.cst_reduction import (FieldMetric, FunctionalDiffusionMap, neighbor_predict,
    CSTLatentDecoder, SmoothQoISurrogate, cst_surfaces, evaluate_cst, operating_weights)
from utils.qoi_fdm import objectives, NAMES

OUT=ROOT/'data/qoi_fdm'; FIG=ROOT/'figures/qoi_fdm'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
plt.style.use(ROOT/'utils/academic.mplstyle')
cache=ROOT/'data/cst_reset/database_b5b0c0e969be0522.npz'
db=np.load(cache); a=db['alphas']; re=db['reynolds']
allq=objectives(db['fields'],a,re)
eligible=(db['confidence'].min((1,2))>=.75)&np.all(np.isfinite(list(allq.values())),axis=0)
ids=np.flatnonzero(eligible); f=db['fields'][ids]; p=db['parameters'][ids]
qs={k:v[ids] for k,v in allq.items()}
x=(1-np.cos(np.linspace(0,np.pi,301)))/2
u,l=cst_surfaces(p,x); shape=np.concatenate([u,l],axis=1)
rows=[]; tuning=[]; opts=[]; inverses=[]; archives={'source_index':ids}
for seed in (42,43,44):
    ix=np.random.default_rng(seed).permutation(len(ids)); nt=int(.65*len(ids)); nv=int(.15*len(ids))
    tr,va,te=ix[:nt],ix[nt:nt+nv],ix[nt+nv:]
    archives[f'train_{seed}']=tr; archives[f'validation_{seed}']=va; archives[f'test_{seed}']=te
    metric=FieldMetric(operating_weights(a,re)).fit(f[tr]); features=metric.transform(f)
    for name,q in qs.items():
        best=None
        for weight in (.25,1.,4.,16.):
            for bandwidth in (.25,1.,4.):
                model=FunctionalDiffusionMap(2,weight,bandwidth).fit(features[tr],q[tr])
                z=model.transform(features,q)
                pred,_=neighbor_predict(z[tr],z[va],q[tr])
                err=np.sqrt(np.mean((pred-q[va])**2))/q[tr].std(ddof=1)
                fp,_=neighbor_predict(z[tr],z[va],f[tr])
                tuning.append(dict(seed=seed,qoi=name,weight=weight,bandwidth=bandwidth,q_error=err,field_error=metric.relative_error(f[va],fp)))
                if best is None or err<best[0]: best=(err,model,z,weight,bandwidth)
        _,model,z,weight,bandwidth=best
        pred,nn=neighbor_predict(z[tr],z[te],q[tr]); fp,_=neighbor_predict(z[tr],z[te],f[tr]); sp,_=neighbor_predict(z[tr],z[te],shape[tr])
        reference=np.argsort(cdist(features[te],features[tr]),axis=1)[:,:5]
        local=np.argsort(cdist(z[te],z[tr]),axis=1)[:,:5]
        recall=np.mean([len(set(i)&set(j))/5 for i,j in zip(reference,local)])
        decoder=CSTLatentDecoder(z[tr],p[tr]); inside=decoder.triangulation.find_simplex((z[te]-decoder.center)/decoder.scale)>=0
        decoded=np.array([decoder.decode(v)[0] for v in z[te][inside]])
        du,dl=cst_surfaces(decoded,x)
        geom=np.sqrt(np.mean((np.concatenate([du,dl],axis=1)-shape[te][inside])**2))
        row=dict(seed=seed,qoi=name,weight=weight,bandwidth=bandwidth,nrmse=np.sqrt(np.mean((pred-q[te])**2))/q[tr].std(ddof=1),rho=spearmanr(pred,q[te]).statistic,field_error=metric.relative_error(f[te],fp),recall5=recall,shape_knn_rmse=np.sqrt(np.mean((sp-shape[te])**2)),shape_mean_rmse=np.sqrt(np.mean((shape[tr].mean(0)-shape[te])**2)),hull_coverage=inside.mean(),shape_inverse_rmse=geom)
        rows.append(row)
        if seed==42:
            archives[name+'_z']=z; archives[name+'_q']=q
            # Re-evaluate a fixed subset of held-out inverse images.
            selected=np.flatnonzero(inside)[:30]; dec=np.array([decoder.decode(v)[0] for v in z[te][selected]])
            df,dc=evaluate_cst(dec,a,re); dq=objectives(df,a,re)[name]
            rz=model.transform(metric.transform(df),dq)
            scale=np.median(pdist(z[tr]))
            for j,k in enumerate(selected):
                inverses.append(dict(qoi=name,source_index=int(ids[te[k]]),q_original=q[te[k]],q_decoded=dq[j],q_error_standardized=(dq[j]-q[te[k]])/q[tr].std(ddof=1),roundtrip=np.linalg.norm(rz[j]-z[te[k]])/scale,confidence=dc[j].min()))
            bests=None
            for ridge in (.001,.01,.1):
                for length in (.5,1.,2.):
                    s=SmoothQoISurrogate(ridge,length).fit(z[tr],q[tr]); e=np.mean((s.predict(z[va])-q[va])**2)
                    if bests is None or e<bests[0]: bests=(e,s)
            candidates=decoder.optimize(bests[1],q[tr],starts=8)
            incumbent=p[tr[np.argmin(q[tr])]]
            cp=[incumbent]; positions=[]
            for c in candidates:
                if c['feasible']:
                    cp.append(decoder.decode(c['coordinates'])[0]); positions.append(c)
            cp=np.array(cp); cf,cc=evaluate_cst(cp,a,re); cq=objectives(cf,a,re)[name]
            vf,vc=evaluate_cst(cp,a,re,'xxxlarge'); vq=objectives(vf,a,re)[name]
            fa=np.linspace(a[0],a[-1],61); fr=np.geomspace(re[0],re[-1],41)
            gf,gc=evaluate_cst(cp,fa,fr); gq=objectives(gf,fa,fr)[name]
            cu,cl=cst_surfaces(cp,x); thick=(cu-cl).max(1); camber=np.abs((cu+cl)/2).max(1)
            zz=model.transform(metric.transform(cf),cq)
            archives[name+'_candidate_parameters']=cp
            for j,c in enumerate(positions,1):
                opts.append(dict(qoi=name,start=j,success=c['success'],q_predicted=c['predicted_q'],q_actual=cq[j],incumbent=cq[0],gain_percent=100*(cq[0]-cq[j])/abs(cq[0]),variant_gain_percent=100*(vq[0]-vq[j])/abs(vq[0]),fine_gain_percent=100*(gq[0]-gq[j])/abs(gq[0]),confidence=cc[j].min(),variant_confidence=vc[j].min(),fine_confidence=gc[j].min(),geometry_valid=bool(.08<=thick[j]<=.20 and camber[j]<=.05 and (cu[j]-cl[j]).min()>=0),roundtrip=np.linalg.norm(zz[j]-c['coordinates'])/scale))
        print(seed,name,round(row['nrmse'],4),flush=True)
        pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
        pd.DataFrame(opts).to_csv(OUT/'optimization.csv',index=False)
        pd.DataFrame(inverses).to_csv(OUT/'inverse.csv',index=False)
        pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False)
np.savez_compressed(OUT/'embeddings.npz',**archives)
metadata=dict(seeds=[42,43,44],n_raw=len(db['fields']),n_confident=int((db['confidence'].min((1,2))>=.75).sum()),n_common=len(ids),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),python=platform.python_version(),versions={k:__import__(k).__version__ for k in ('numpy','scipy','pandas','matplotlib','neuralfoil')},source_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ('utils/qoi_fdm.py','utils/cst_reduction.py','notebooks/qoi_fdm.py')})
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
m=pd.DataFrame(rows); summary=m.groupby('qoi').mean(numeric_only=True).reindex(NAMES); summary.to_csv(OUT/'summary.csv')
fig,axes=plt.subplots(4,3,figsize=(10,11))
for ax,name in zip(axes.flat,NAMES):
    z=archives[name+'_z']; q=qs[name]; sc=ax.scatter(z[:,0],z[:,1],c=q,s=5,cmap='viridis'); ax.set_title(name.replace('_',' ')); ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$'); fig.colorbar(sc,ax=ax,shrink=.8)
fig.tight_layout(); fig.savefig(FIG/'embeddings.pdf'); plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,4.5))
for ax,col,title in zip(axes,['nrmse','field_error','shape_inverse_rmse'],['Conditional QoI NRMSE','Field relative squared error','Inverse shape RMS / chord']):
    for j,name in enumerate(NAMES):
        values=m[m.qoi==name][col]; ax.plot(values,[j]*len(values),'o',ms=3)
    ax.set_yticks(range(len(NAMES)),[n.replace('_',' ') for n in NAMES] if ax==axes[0] else []); ax.set_xlabel(title); ax.grid(axis='x',alpha=.3)
fig.tight_layout(); fig.savefig(FIG/'metrics.pdf'); plt.close(fig)
corr=pd.DataFrame(qs).corr(method='spearman'); corr.to_csv(OUT/'objective_correlations.csv')
fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); ax.set_xticks(range(12),NAMES,rotation=65,ha='right'); ax.set_yticks(range(12),NAMES); fig.colorbar(im,ax=ax,label='Spearman correlation'); fig.tight_layout(); fig.savefig(FIG/'correlations.pdf'); plt.close(fig)
print(summary.to_string(),flush=True)
