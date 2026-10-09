"""New CST dataset and quantity-aware functional reduction experiment.

Run from the root: python notebooks/cst_reset.py
Change --qoi or --metric to regenerate the analysis under a different objective.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cst_reduction import (TE_THICKNESS, CSTLatentDecoder, FieldMetric,
    FunctionalDiffusionMap, SmoothQoISurrogate, cst_surfaces, evaluate_cst,
    neighbor_predict, operating_weights, quantity_of_interest, sample_cst)
from utils.functional_clustering import trapezoid_weights

SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_database():
    parameters, sample_index, screen = sample_cst(768, SEED)
    alpha, reynolds = np.linspace(-5, 12, 31), np.geomspace(3e5, 3e6, 21)
    config = dict(seed=SEED, samples=768, independent_dimensions=15, polynomial_degree=7,
        N1=.5, N2=1., TE_thickness=TE_THICKNESS, leading_edge_weight=0.,
        half_thickness_bounds=[.10,.30], camber_weight_bounds=[-.06,.10], b0=0.,
        screen=screen, alphas=alpha.tolist(), reynolds=reynolds.tolist(),
        measure="uniform alpha and log(Re)", coefficients=["CD", "CL", "CM"],
        solver="NeuralFoil", model_size="large", mach=0., n_crit=9.,
        xtr_upper=1., xtr_lower=1.,
        versions={p: version(p) for p in ("NeuralFoil", "AeroSandbox", "numpy", "scipy")})
    encoded = json.dumps(config, sort_keys=True)
    key = hashlib.sha256(encoded.encode()+parameters.tobytes()).hexdigest()[:16]
    directory = ROOT/"data/cst_reset"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f"database_{key}.npz"
    if path.exists():
        with np.load(path) as z:
            if str(z["config"]) != encoded:
                raise ValueError("Cache configuration mismatch.")
            np.testing.assert_array_equal(parameters, z["parameters"])
            fields, confidence = z["fields"], z["confidence"]
        print(f"Reusing {path.name}", flush=True)
    else:
        fields, confidence = [], []
        started = time.perf_counter()
        for begin in range(0, len(parameters), 64):
            f, c = evaluate_cst(parameters[begin:begin+64], alpha, reynolds, config["model_size"])
            fields.append(f)
            confidence.append(c)
            print(f"CST fields: {min(begin+64,len(parameters))}/{len(parameters)}; "
                  f"{time.perf_counter()-started:.1f} s", flush=True)
        fields, confidence = np.concatenate(fields), np.concatenate(confidence)
        np.savez_compressed(path, parameters=parameters, sobol_index=sample_index,
                            alphas=alpha, reynolds=reynolds, fields=fields,
                            confidence=confidence, coefficients=np.array(config["coefficients"]),
                            config=encoded)
    if not np.isfinite(fields).all() or not np.isfinite(confidence).all() or np.any(fields[...,0] <= 0):
        raise ValueError("Invalid persisted fields or confidence.")
    return path, config, parameters, fields, confidence, alpha, reynolds


def recall(reference, projected, train_index, query_index, neighbors=5):
    source = np.argsort(cdist(reference[query_index], reference[train_index]), axis=1)[:, :neighbors]
    latent = np.argsort(cdist(projected[query_index], projected[train_index]), axis=1)[:, :neighbors]
    return float(np.mean([len(set(a)&set(b))/neighbors for a,b in zip(source, latent)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qoi", choices=("min_drag", "mean_drag", "negative_efficiency"), default="min_drag")
    parser.add_argument("--metric", choices=("balanced", "drag"), default="balanced")
    args = parser.parse_args()
    plt.style.use(ROOT/"utils/academic.mplstyle")
    cache, config, all_parameters, all_fields, all_confidence, alpha, reynolds = load_database()
    run_name = f"{args.qoi}_{args.metric}"
    data, figures = ROOT/"data/cst_reset"/run_name, ROOT/"figures/cst_reset"/run_name
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    # The full raw corpus is preserved; the analysis cohort has a declared confidence screen.
    eligible = all_confidence.min(axis=(1,2)) >= .75
    source_index = np.flatnonzero(eligible)
    parameters, fields, confidence = all_parameters[eligible], all_fields[eligible], all_confidence[eligible]
    if len(fields) < 200:
        raise ValueError("Too few confidence-screened fields for the planned split.")
    rng = np.random.default_rng(SEED)
    permutation = rng.permutation(len(fields))
    n_train, n_valid = int(.65*len(fields)), int(.15*len(fields))
    train, valid, test = permutation[:n_train], permutation[n_train:n_train+n_valid], permutation[n_train+n_valid:]
    split = np.full(len(fields), "test", dtype="U10")
    split[train], split[valid] = "train", "validation"
    print(f"Analysis cohort {len(fields)}/768; splits {len(train)}/{len(valid)}/{len(test)}", flush=True)
    weights = operating_weights(alpha, reynolds)
    q = quantity_of_interest(fields, weights, args.qoi)
    metric = FieldMetric(weights, args.metric).fit(fields[train])
    features = metric.transform(fields)
    x = (1-np.cos(np.linspace(0,np.pi,301)))/2
    upper, lower = cst_surfaces(parameters, x)
    shape_features = (np.stack([upper,lower],axis=-1)*np.sqrt(trapezoid_weights(x)[None,:,None]/2)).reshape(len(fields),-1)
    thickness = (upper-lower).max(axis=1)
    signed_camber = ((upper+lower)/2)[np.arange(len(fields)), np.argmax(np.abs((upper+lower)/2),axis=1)]
    pca = PCA(n_components=min(30,len(train)),svd_solver="full").fit(features[train])
    pca_coordinates = pca.transform(features)
    tuning, selected, embeddings, models = [], {}, {}, {}
    for dimensions in (2,3):
        embeddings[f"FPCA-{dimensions}"] = pca_coordinates[:,:dimensions]
        models[f"FPCA-{dimensions}"] = None
        for method, q_weights in (("FDM", (0.,)), ("QoI-FDM", (.1,.25,1.,4.,16.))):
            best = None
            for q_weight in q_weights:
                for bandwidth in (.1,.25,1.,4.,16.):
                    fitted = FunctionalDiffusionMap(dimensions, q_weight, bandwidth).fit(features[train],q[train])
                    z = fitted.transform(features, q)
                    np.testing.assert_allclose(z[train], fitted.coordinates, atol=1e-9)
                    predicted, neighbors = neighbor_predict(z[train], z[valid],q[train])
                    q_error = float(np.sqrt(np.mean((predicted-q[valid])**2))/np.std(q[train],ddof=1))
                    reconstructed, _ = neighbor_predict(z[train],z[valid],fields[train])
                    field_error = metric.relative_error(fields[valid], reconstructed)
                    # FDM control selects on fields; QoI-FDM selects on the stated objective.
                    score = field_error if method == "FDM" else q_error
                    row = dict(method=method, dimensions=dimensions, q_weight=q_weight,
                               bandwidth=bandwidth, validation_q_nrmse=q_error,
                               validation_field_error=field_error, selection_score=score,
                               min_participation_fraction=float(np.min(
                                   np.sum(fitted.coordinates**2,axis=0)**2/
                                   np.sum(fitted.coordinates**4,axis=0)/len(train))))
                    tuning.append(row)
                    if best is None or score < best[0]:
                        best = score, fitted, z, row
            name = f"{method}-{dimensions}"
            models[name], embeddings[name], selected[name] = best[1], best[2], best[3]
            print(f"Selected {name}: {best[3]}",flush=True)
    rows, surrogate_tuning, surrogates, cluster_rows, archive = [], [], {}, [], {}
    assignments = pd.DataFrame(dict(source_index=source_index, split=split,
        q=q, thickness=thickness, camber=signed_camber,
        confidence_min=confidence.min(axis=(1,2))))
    for name, z in embeddings.items():
        predicted, neighbors = neighbor_predict(z[train],z[test],q[train])
        field_prediction, _ = neighbor_predict(z[train],z[test],fields[train])
        geometry_prediction, _ = neighbor_predict(z[train],z[test],shape_features[train])
        q_spread = np.std(q[train][neighbors],axis=1,ddof=1).mean()/q[train].std(ddof=1)
        best = None
        for ridge in (.001,.01,.1):
            for length in (.5,1.,2.):
                surrogate = SmoothQoISurrogate(ridge,length).fit(z[train],q[train])
                validation_rmse = float(np.sqrt(np.mean((surrogate.predict(z[valid])-q[valid])**2)))
                surrogate_tuning.append(dict(method=name,ridge=ridge,length_scale=length,validation_rmse=validation_rmse))
                if best is None or validation_rmse < best[0]:
                    best = validation_rmse, surrogate, ridge, length
        surrogates[name] = best[1]
        smooth_prediction = best[1].predict(z[test])
        row = dict(method=name, dimensions=z.shape[1], test_n=len(test),
            q_neighbor_nrmse=float(np.sqrt(np.mean((predicted-q[test])**2))/q[train].std(ddof=1)),
            q_smooth_nrmse=float(np.sqrt(np.mean((smooth_prediction-q[test])**2))/q[train].std(ddof=1)),
            q_smooth_spearman=float(spearmanr(smooth_prediction,q[test]).statistic),
            q_neighbor_spread=float(q_spread), field_relative_error=metric.relative_error(fields[test],field_prediction),
            field_recall5=recall(features,z,train,test), shape_recall5=recall(shape_features,z,train,test),
            shape_rmse=float(np.sqrt(np.mean(np.sum((geometry_prediction-shape_features[test])**2,axis=1)))),
            surrogate_ridge=best[2],surrogate_length=best[3])
        # Direct linear reconstruction is meaningful only for FPCA and the fitted field metric.
        if name.startswith("FPCA"):
            fhat = z[test] @ pca.components_[:z.shape[1]]+pca.mean_
            row["linear_field_relative_error"] = float(np.sum((features[test]-fhat)**2)/np.sum(features[test]**2))
            row["training_variance_retained"] = float(pca.explained_variance_ratio_[:z.shape[1]].sum())
        rows.append(row)
        for axis in range(z.shape[1]):
            assignments[f"{name}_z{axis+1}"] = z[:,axis]
        clustering = KMeans(n_clusters=4,n_init=20,random_state=SEED).fit(z[train])
        labels = clustering.predict(z)
        assignments[f"{name}_cluster"] = labels
        for k in range(4):
            members = labels == k
            cluster_rows.append(dict(method=name, cluster=k, n=int(members.sum()),
                q_mean=float(q[members].mean()),q_std=float(q[members].std(ddof=0)),
                thickness_mean=float(thickness[members].mean()),camber_mean=float(signed_camber[members].mean())))
        archive[f"{name}_coordinates"] = z
        archive[f"{name}_smooth_test_prediction"] = smooth_prediction
        if models[name] is not None:
            for attr in ("eigenvalues","eigenfunctions","density","degree"):
                archive[f"{name}_{attr}"] = getattr(models[name],attr)
    results = pd.DataFrame(rows)
    results.to_csv(data/"projection_metrics.csv", index=False)
    pd.DataFrame(tuning).to_csv(data/"embedding_tuning.csv",index=False)
    pd.DataFrame(surrogate_tuning).to_csv(data/"surrogate_tuning.csv",index=False)
    pd.DataFrame(cluster_rows).to_csv(data/"cluster_summary.csv",index=False)
    assignments.to_csv(data/"assignments.csv",index=False)
    np.savez_compressed(data/"fitted_reductions.npz", train=train,validation=valid,test=test,
        source_index=source_index,q=q,mean=metric.mean,scales=metric.scales,weights=weights,
        pca_components=pca.components_,pca_variance=pca.explained_variance_ratio_,**archive)
    optimize_rows, optimization_arrays = optimize_embeddings(embeddings,surrogates,models,
        parameters,fields,q,train,metric,alpha,reynolds,args.qoi,pca)
    optimization = pd.DataFrame(optimize_rows)
    optimization.to_csv(data/"optimization.csv",index=False)
    adopted=[]
    baseline=int(train[np.argmin(q[train])])
    for name in embeddings:
        candidates=optimization[(optimization.method==name)&optimization.geometry_ok&optimization.confidence_ok]
        best=candidates.loc[candidates.actual_q.idxmin()] if len(candidates) else None
        use_candidate=best is not None and best.actual_q < q[baseline]
        if use_candidate:
            p=optimization_arrays[f"{name}_parameters"][int(best.candidate)]
            value=float(best.actual_q)
        else:
            p,value=parameters[baseline],float(q[baseline])
        optimization_arrays[f"{name}_adopted_parameters"]=p
        adopted.append(dict(method=name,source="candidate" if use_candidate else "training incumbent",
                            q=value,baseline_source_index=int(source_index[baseline])))
    pd.DataFrame(adopted).to_csv(data/"adopted_designs.csv",index=False)
    np.savez_compressed(data/"optimization_candidates.npz",**optimization_arrays)
    # A finer-grid QoI check on fixed, preselected training shapes, not new fitting data.
    check_indices = train[np.argsort(q[train])[np.linspace(0,len(train)-1,12,dtype=int)]]
    finer_a, finer_r = np.linspace(alpha[0],alpha[-1],61),np.geomspace(reynolds[0],reynolds[-1],41)
    finer_fields,finer_conf = evaluate_cst(parameters[check_indices],finer_a,finer_r)
    finer_q = quantity_of_interest(finer_fields,operating_weights(finer_a,finer_r),args.qoi)
    pd.DataFrame(dict(source_index=source_index[check_indices],coarse_q=q[check_indices],
                      fine_q=finer_q,relative_change=(finer_q-q[check_indices])/np.abs(q[check_indices]),
                      fine_confidence_min=finer_conf.min(axis=(1,2)))).to_csv(data/"grid_check.csv",index=False)
    minimum_index = np.argmin(fields[...,0].reshape(len(fields),-1),axis=1)
    ai,ri = np.unravel_index(minimum_index,fields.shape[1:3])
    corpus = pd.DataFrame(dict(source_index=np.arange(len(all_fields)), eligible=eligible,
        confidence_min=all_confidence.min(axis=(1,2)),
        fraction_confidence_below075=(all_confidence < .75).mean(axis=(1,2))))
    corpus.to_csv(data/"corpus_audit.csv",index=False)
    all_u,all_l=cst_surfaces(all_parameters,x)
    catalogue=pd.DataFrame(all_parameters,columns=[f"{side}_{i}" for side in ("upper","lower") for i in range(8)])
    catalogue.insert(0,"source_index",np.arange(len(all_parameters)))
    catalogue["max_thickness"]=(all_u-all_l).max(axis=1)
    catalogue["max_abs_camber"]=np.abs((all_u+all_l)/2).max(axis=1)
    catalogue["confidence_min"]=all_confidence.min(axis=(1,2))
    catalogue["eligible"]=eligible
    catalogue["min_drag"]=quantity_of_interest(all_fields,weights,"min_drag")
    catalogue["mean_drag"]=quantity_of_interest(all_fields,weights,"mean_drag")
    catalogue.to_csv(ROOT/"data/cst_reset/catalogue.csv",index=False)
    # Solver-size sensitivity is not validation against independent physics.
    probes=list(parameters[check_indices])
    origins=[f"training-{int(source_index[i])}" for i in check_indices]
    large_q=list(q[check_indices])
    for method in embeddings:
        group=optimization[(optimization.method==method)&optimization.geometry_ok&optimization.confidence_ok]
        if len(group):
            best=group.loc[group.actual_q.idxmin()]
            probes.append(optimization_arrays[f"{method}_parameters"][int(best.candidate)])
            origins.append(method)
            large_q.append(float(best.actual_q))
    variant_fields,variant_conf=evaluate_cst(np.asarray(probes),alpha,reynolds,model_size="xxxlarge")
    variant_q=quantity_of_interest(variant_fields,weights,args.qoi)
    pd.DataFrame(dict(origin=origins,q_large=large_q,q_xxxlarge=variant_q,
        relative_change=(variant_q-np.asarray(large_q))/np.abs(large_q),
        confidence_min_xxxlarge=variant_conf.min(axis=(1,2)))).to_csv(data/"model_size_check.csv",index=False)
    summary = dict(question="Can QoI-aware functional coordinates preserve minimum-drag variation and support feasible latent optimization?",
        qoi=args.qoi, metric=args.metric, corpus_n=len(all_fields),cohort_n=len(fields),
        excluded_n=int((~eligible).sum()),split_counts=dict(train=len(train),validation=len(valid),test=len(test)),
        confidence_threshold=.75, q_range=[float(q.min()),float(q.max())],
        min_drag_highest_re_fraction=float(np.mean(ri==len(reynolds)-1)),
        min_drag_alpha_boundary_fraction=float(np.mean((ai==0)|(ai==len(alpha)-1))),
        max_grid_relative_change=float(np.max(np.abs((finer_q-q[check_indices])/q[check_indices]))),
        max_model_size_relative_change=float(np.max(np.abs((variant_q-np.asarray(large_q))/np.asarray(large_q)))),
        training_variance_2=float(pca.explained_variance_ratio_[:2].sum()),
        training_variance_3=float(pca.explained_variance_ratio_[:3].sum()),
        training_rank99=int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_),.99)+1),
        chosen_embeddings=selected,projection_metrics=rows,
        best_training_q=float(q[train].min()),best_corpus_q=float(q.min()),
        optimization=optimize_rows)
    (data/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,
        python=platform.python_version(), versions={p:version(p) for p in ("numpy","scipy","pandas","matplotlib","scikit-learn","NeuralFoil","AeroSandbox")},
        database=str(cache.relative_to(ROOT)),database_sha256=sha256(cache),generation=config,
        specification_sha256=sha256(ROOT/"next_experimentations.md"),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),ROOT/"utils/cst_reduction.py",ROOT/"utils/functional_clustering.py")},
        qoi=args.qoi,metric=args.metric,selected_embeddings=selected,
        cohort_rule="min pointwise NeuralFoil confidence >= 0.75",
        evaluation="conditional projection of held-out fields; QoI supplied to QoI-FDM extension",
        optimization="training-only smooth QoI surrogate; eight hull-constrained starts; CST barycentric decoder; fresh NeuralFoil evaluation")
    (data/"run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    make_figures(figures,parameters,fields,confidence,alpha,reynolds,q,thickness,signed_camber,
                 embeddings,results,test,summary,assignments)
    make_tradeoff_figure(figures,pd.DataFrame(tuning))
    write_report(summary,results,figures,run_name)
    print(results.round(4).to_string(index=False),flush=True)
    print(optimization[["method","actual_q","predicted_q","confidence_min","roundtrip_relative"]].to_string(index=False),flush=True)


def optimize_embeddings(embeddings,surrogates,models,parameters,fields,q,train,metric,alpha,reynolds,qoi,pca):
    rows, archive = [], {}
    for name,z in embeddings.items():
        decoder = CSTLatentDecoder(z[train],parameters[train])
        solves = decoder.optimize(surrogates[name],q[train],starts=8,seed=SEED)
        usable = sorted([r for r in solves if r["feasible"]],key=lambda r:r["predicted_q"])
        # Re-evaluate at most three distinct predicted minima; save every solve status.
        candidates = []
        for r in usable:
            decoded = decoder.decode(r["coordinates"])[0]
            if not any(np.linalg.norm(decoded-decoder.decode(c["coordinates"])[0]) < 1e-4 for c in candidates):
                candidates.append(r)
            if len(candidates) == 3:
                break
        if not candidates:
            raise ValueError(f"No feasible optimizer candidate for {name}.")
        p = np.stack([decoder.decode(r["coordinates"])[0] for r in candidates])
        f,c = evaluate_cst(p,alpha,reynolds)
        actual = quantity_of_interest(f,metric.weights,qoi)
        fine_a,fine_r = np.linspace(alpha[0],alpha[-1],61),np.geomspace(reynolds[0],reynolds[-1],41)
        fine_f,fine_c = evaluate_cst(p,fine_a,fine_r)
        fine_q = quantity_of_interest(fine_f,operating_weights(fine_a,fine_r),qoi)
        x = np.linspace(0,1,1001)
        upper,lower = cst_surfaces(p,x)
        max_t,max_c = (upper-lower).max(axis=1),np.abs((upper+lower)/2).max(axis=1)
        geometry_ok = (max_t>=.08-1e-8)&(max_t<=.20+1e-8)&(max_c<=.05+1e-8)&np.all(upper>=lower,axis=1)
        for i,r in enumerate(candidates):
            field_features = metric.transform(f[i:i+1])
            if models[name] is None:
                new_z = pca.transform(field_features)[0,:z.shape[1]]
            else:
                new_z = models[name].transform(field_features,actual[i:i+1])[0]
            typical = np.median(cdist(z[train],z[train])[np.triu_indices(len(train),1)])
            rows.append(dict(method=name,candidate=i,predicted_q=r["predicted_q"],actual_q=float(actual[i]),
                improvement_over_training_pct=float(100*(q[train].min()-actual[i])/abs(q[train].min())),
                confidence_min=float(c[i].min()),geometry_ok=bool(geometry_ok[i]),
                fine_q=float(fine_q[i]),fine_confidence_min=float(fine_c[i].min()),
                confidence_ok=bool(c[i].min()>=.75),roundtrip_relative=float(np.linalg.norm(new_z-r["coordinates"])/typical),
                optimizer_success=r["success"],optimizer_iterations=r["iterations"],
                optimizer_message=r["message"], attempted_starts=len(solves),
                failed_starts=sum(not s["success"] for s in solves)))
        archive[f"{name}_parameters"], archive[f"{name}_fields"], archive[f"{name}_confidence"] = p,f,c
        archive[f"{name}_coordinates"] = np.stack([r["coordinates"] for r in candidates])
        archive[f"{name}_all_solve_coordinates"] = np.stack([r["coordinates"] for r in solves])
        archive[f"{name}_all_solve_status"] = np.array([r["message"] for r in solves])
        archive[f"{name}_all_solve_success"] = np.array([r["success"] for r in solves])
        archive[f"{name}_all_solve_feasible"] = np.array([r["feasible"] for r in solves])
        archive[f"{name}_all_solve_iterations"] = np.array([r["iterations"] for r in solves])
    return rows,archive


def make_figures(figures,parameters,fields,confidence,alpha,reynolds,q,thickness,camber,
                 embeddings,results,test,summary,assignments):
    def save(fig,name):
        fig.savefig(figures/f"{name}.pdf")
        fig.savefig(figures/f"{name}.png",dpi=180)
        plt.close(fig)
    selected = np.argsort(thickness)[np.linspace(0,len(q)-1,6,dtype=int)]
    x = np.linspace(0,1,301)
    upper,lower = cst_surfaces(parameters[selected],x)
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.8),layout="constrained")
    for j,i in enumerate(selected):
        line,=axes[0].plot(x,upper[j],label=f"t/c={thickness[i]:.3f}")
        axes[0].plot(x,lower[j],color=line.get_color())
        axes[1].plot(alpha,fields[i,:,-1,0],color=line.get_color())
    axes[0].set(xlabel="x/c",ylabel="y/c",title="CST sample contours")
    axes[0].set_aspect("equal",adjustable="datalim")
    axes[0].legend(fontsize=6,ncol=2)
    axes[1].set(xlabel=r"$\alpha$ (deg)",ylabel=r"$C_D$",title=r"Drag at $Re=3\times10^6$")
    save(fig,"dataset")
    fig,axes = plt.subplots(3,3,figsize=(7.2,6.2),layout="constrained")
    for col,method in enumerate(("FPCA-2","FDM-2","QoI-FDM-2")):
        z = embeddings[method]
        for row,(values,label) in enumerate(((q,"QoI (minimize)"),(100*thickness,"Max thickness (% chord)"),(100*camber,"Signed max camber (% chord)"))):
            ax=axes[row,col]
            scatter=ax.scatter(*z.T,c=values,s=7,cmap="viridis",vmin=values.min(),vmax=values.max())
            ax.set(xlabel="Coordinate 1",ylabel="Coordinate 2",title=method if row==0 else None)
            if col==2:
                fig.colorbar(scatter,ax=axes[row,:],label=label,shrink=.8)
    save(fig,"projection_maps")
    fig=plt.figure(figsize=(7.2,2.7),layout="constrained")
    for j,method in enumerate(("FPCA-3","FDM-3","QoI-FDM-3")):
        ax=fig.add_subplot(1,3,j+1,projection="3d")
        ax.scatter(*embeddings[method].T,c=q,s=6,cmap="viridis",vmin=q.min(),vmax=q.max())
        ax.set(title=method,xlabel="z1",ylabel="z2",zlabel="z3")
        ax.tick_params(labelsize=6)
    save(fig,"projection_3d")
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout="constrained")
    for ax,method in zip(axes,("FPCA-2","FDM-2","QoI-FDM-2")):
        ax.scatter(*embeddings[method].T,c=assignments[f"{method}_cluster"],cmap="tab10",s=8,vmin=0,vmax=9)
        ax.set(title=method,xlabel="z1",ylabel="z2")
    save(fig,"clusters")
    fig,axes=plt.subplots(1,3,figsize=(7.2,3),layout="constrained")
    for ax,column,label in zip(axes,("q_smooth_nrmse","field_relative_error","field_recall5"),
                               ("Smooth QoI NRMSE","Field reconstruction error","Field five-neighbor recall")):
        for method in ("FPCA","FDM","QoI-FDM"):
            group=results[results.method.str.startswith(method+"-")].sort_values("dimensions")
            ax.plot(group.dimensions,group[column],"o-",label=method)
        ax.set(xlabel="Dimensions",ylabel=label,xticks=[2,3])
    axes[-1].legend(fontsize=7)
    save(fig,"preservation")
    fig,ax=plt.subplots(figsize=(7.2,3),layout="constrained")
    optimization=pd.DataFrame(summary["optimization"])
    for j,method in enumerate(embeddings):
        group=optimization[optimization.method==method]
        ax.scatter(np.full(len(group),j)-.08,group.predicted_q,marker="x",color="tab:orange")
        ax.scatter(np.full(len(group),j)+.08,group.actual_q,marker="o",color="tab:blue")
    ax.scatter([],[],marker="x",color="tab:orange",label="Latent surrogate")
    ax.scatter([],[],marker="o",color="tab:blue",label="Decoded CST / NeuralFoil")
    ax.axhline(summary["best_training_q"],color=".3",ls="--",label="Best training shape")
    ax.set(xticks=range(len(embeddings)),xticklabels=list(embeddings),ylabel="QoI (lower is better)")
    ax.tick_params(axis="x",rotation=20)
    ax.legend(fontsize=7)
    save(fig,"optimization")


def write_report(summary,results,figures,run_name):
    # The report body is authored after inspecting the first completed run.
    report=ROOT/"report"
    report.mkdir(exist_ok=True)
    lines=[r"\begin{tabular}{lrrrrr}",r"\toprule",
           r"Method & QoI NRMSE & QoI $\rho$ & Field error & Field recall & Shape recall\\",r"\midrule"]
    for _,r in results.iterrows():
        lines.append(f"{r.method} & {r.q_smooth_nrmse:.3f} & {r.q_smooth_spearman:.3f} & {r.field_relative_error:.3f} & {r.field_recall5:.3f} & {r.shape_recall5:.3f}"+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (report/f"cst_{run_name}_table.tex").write_text("\n".join(lines)+"\n")
    lines=[r"\begin{tabular}{lrrrrr}",r"\toprule",
           r"Method & Predicted $q$ & Actual $q$ & Fine-grid $q$ & Change (\%) & Round trip\\",r"\midrule"]
    optimization=pd.DataFrame(summary["optimization"])
    for name in results.method:
        group=optimization[(optimization.method==name)&optimization.geometry_ok&optimization.confidence_ok]
        row=group.loc[group.actual_q.idxmin()]
        lines.append(f"{name} & {row.predicted_q:.6f} & {row.actual_q:.6f} & {row.fine_q:.6f} & "
                     f"{-row.improvement_over_training_pct:+.2f} & {row.roundtrip_relative:.3f}"+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (report/f"cst_{run_name}_optimization_table.tex").write_text("\n".join(lines)+"\n")


def make_tradeoff_figure(figures,tuning):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout="constrained",sharey=True)
    for ax,dimensions in zip(axes,(2,3)):
        rows=tuning[tuning.dimensions==dimensions]
        for weight,group in rows.groupby("q_weight"):
            ax.scatter(group.validation_field_error,group.validation_q_nrmse,
                       label=f"QoI weight {weight:g}",s=22)
        ax.set(xlabel="Validation field error",ylabel="Validation QoI neighbor NRMSE",
               title=f"{dimensions} dimensions",yscale="log")
    axes[-1].legend(fontsize=6,ncol=2)
    fig.savefig(figures/"metric_tradeoff.pdf")
    fig.savefig(figures/"metric_tradeoff.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
