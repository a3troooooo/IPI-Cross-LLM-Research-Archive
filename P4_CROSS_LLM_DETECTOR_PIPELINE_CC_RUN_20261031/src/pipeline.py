
from __future__ import annotations
import json, hashlib, os, platform, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sklearn
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CFG = json.loads((ROOT/"config/experiment_config.json").read_text())
VEXP = json.loads((ROOT/"config/V_exp_manifest.json").read_text())
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
MODELS_DIR = OUT / "models"; MODELS_DIR.mkdir(exist_ok=True)
DEV_DIR = OUT / "development"; DEV_DIR.mkdir(exist_ok=True)
LOCK_DIR = OUT / "locked_test_scores"; LOCK_DIR.mkdir(exist_ok=True)

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def load_data():
    f77=pd.read_csv(DATA/"features_p3_77_1131.csv")
    f69=pd.read_csv(DATA/"features_p4_69_1131.csv")
    meta=pd.read_csv(DATA/"metadata_membership_1131.csv")
    lab=pd.read_csv(DATA/"train_validation_labels_897.csv")
    for d in (f77,f69,meta,lab):
        d["trajectory_id"]=d["trajectory_id"].astype(str)
    if f77.trajectory_id.duplicated().any() or f69.trajectory_id.duplicated().any() or meta.trajectory_id.duplicated().any():
        raise ValueError("Duplicate trajectory_id detected.")
    if set(f77.trajectory_id)!=set(meta.trajectory_id) or set(f69.trajectory_id)!=set(meta.trajectory_id):
        raise ValueError("Feature/membership trajectory IDs differ.")
    if set(lab.trajectory_id) != set(meta.loc[meta.split.isin(["train","validation"]),"trajectory_id"]):
        raise ValueError("Train/validation label IDs do not match frozen membership.")
    if not set(lab.ipi_affected.unique()).issubset({0,1}):
        raise ValueError("Unexpected label values.")
    # Verify supplied 77 and 69 matrices agree on their shared columns.
    shared=[c for c in f69.columns if c!="trajectory_id"]
    a=f77.set_index("trajectory_id").sort_index()[shared]
    b=f69.set_index("trajectory_id").sort_index()[shared]
    if not a.equals(b):
        if not np.allclose(a.to_numpy(dtype=float),b.to_numpy(dtype=float),equal_nan=True):
            raise ValueError("V77 and V69 shared feature values do not match.")
    return f77, f69, meta, lab

def get_feature_cols(df, version, regime):
    cols=[c for c in df.columns if c!="trajectory_id"]
    if regime=="V_exp":
        missing=set(VEXP["features"])-set(cols)
        if missing: raise ValueError(f"V_exp features missing from {version}: {sorted(missing)}")
        cols=list(VEXP["features"])
    forbidden={"target_model","split","suite","condition_group_id","user_task_group_id",
               "trajectory_id","ipi_affected","label_source","original_review_tier","original_review_status",
               "run_id","scenario_type","strict_success"}
    if set(cols)&forbidden: raise ValueError("Forbidden predictor detected.")
    return cols

def build_pipeline(algo):
    if algo=="LR":
        return Pipeline([("imputer",SimpleImputer(strategy="median")),
                         ("scale",StandardScaler()),
                         ("model",LogisticRegression(max_iter=3000,solver="liblinear",random_state=CFG["seed"]))])
    if algo=="RF":
        return Pipeline([("imputer",SimpleImputer(strategy="median")),
                         ("model",RandomForestClassifier(random_state=CFG["seed"],n_jobs=1))])
    if algo=="GB":
        return Pipeline([("imputer",SimpleImputer(strategy="median")),
                         ("model",GradientBoostingClassifier(random_state=CFG["seed"]))])
    raise ValueError(algo)

def search_space(algo):
    if algo=="LR":
        return {"model__C":np.logspace(-3,2,30).tolist(),
                "model__class_weight":[None,"balanced"]}
    if algo=="RF":
        return {"model__n_estimators":[200,400,600],
                "model__max_depth":[None,3,5,8,12],
                "model__min_samples_leaf":[1,2,4,8],
                "model__max_features":["sqrt",0.7,1.0],
                "model__class_weight":[None,"balanced","balanced_subsample"]}
    return {"model__n_estimators":[50,100,150,250],
            "model__learning_rate":[0.02,0.05,0.1],
            "model__max_depth":[1,2,3],
            "model__min_samples_leaf":[2,4,8],
            "model__subsample":[0.7,0.85,1.0]}

def make_groups(frame):
    """
    Build connected-component groups using both frozen P4 group keys.

    Rows sharing either condition_group_id or user_task_group_id
    are connected. Transitive relationships are merged as well.
    """
    required = ["condition_group_id", "user_task_group_id"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing grouping columns: {missing}")

    parent = list(range(len(frame)))
    rank = [0] * len(frame)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    seen = {}

    for i, row in enumerate(
        frame[required].itertuples(index=False, name=None)
    ):
        for prefix, value in zip(("C", "U"), row):
            if pd.isna(value):
                continue

            key = (prefix, str(value))
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    roots = [find(i) for i in range(len(frame))]
    return pd.Series(roots, index=frame.index, name="component_group")

def assert_group_disjoint(train, val):
    for col in ["condition_group_id","user_task_group_id"]:
        x=set(train[col].dropna().astype(str)); y=set(val[col].dropna().astype(str))
        if x&y: raise ValueError(f"Group leakage across Train/Validation in {col}: {len(x&y)} overlaps.")

def threshold_by_balanced_accuracy(y, scores):
    grid=np.unique(np.quantile(scores, np.linspace(0,1,CFG["threshold_grid_size"])))
    if len(grid)==0: return 0.5
    best=(-1,None)
    for t in grid:
        pred=(scores>=t).astype(int)
        b=balanced_accuracy_score(y,pred)
        if b>best[0]: best=(b,float(t))
    return best[1]

def metrics(y, scores, threshold):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float)
    pred=(scores>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"n":int(len(y)),"positive":int(y.sum()),"negative":int((y==0).sum()),
        "prevalence":float(y.mean()) if len(y) else None,
        "auroc":float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,
        "average_precision":float(average_precision_score(y,scores)) if len(np.unique(y))==2 else None,
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall_tpr":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "fpr_all_negative":float(fp/(fp+tn)) if fp+tn else None,
        "fnr":float(fn/(fn+tp)) if fn+tp else None,
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "threshold":float(threshold)}

def experiment_id(version, regime, algo, strategy, source):
    return f"P4_{version}_{regime}_{algo}_{strategy}_{source}_v1"

def train_one(Xtr,ytr,gtr,Xval,yval,algo):
    # GroupKFold on Train only; group labels are composite frozen identifiers.
    n_groups=len(pd.Series(gtr).unique())
    n_splits=min(CFG["cv_splits"],n_groups)
    if n_splits<2: raise ValueError("Insufficient unique groups for group CV.")
    cv=GroupKFold(n_splits=n_splits)
    pipe=build_pipeline(algo)
    search=RandomizedSearchCV(pipe,search_space(algo),n_iter=CFG["random_search_iterations"],
        scoring="average_precision",cv=cv,random_state=CFG["seed"],n_jobs=CFG["n_jobs"],
        refit=True,verbose=0,error_score="raise")
    search.fit(Xtr,ytr,groups=gtr)
    val_scores=search.best_estimator_.predict_proba(Xval)[:,1]
    threshold=threshold_by_balanced_accuracy(yval,val_scores)
    return search, val_scores, threshold

def save_dev_artifacts():
    f77,f69,meta,lab=load_data()
    rows=[]; search_rows=[]; importance_rows=[]
    feature_files={"V77":f77,"V69":f69}
    for version, feat in feature_files.items():
      feat=feat.set_index("trajectory_id")
      for regime in CFG["feature_regimes"]:
       cols=get_feature_cols(feat.reset_index(),version,regime)
       for algo in CFG["algorithms"]:
        # Single-source
        for source in CFG["models"]:
          train_lab=lab[(lab.target_model==source)&(lab.split=="train")].copy()
          val_lab=lab[(lab.target_model==source)&(lab.split=="validation")].copy()
          assert_group_disjoint(train_lab,val_lab)
          train_ids=train_lab.trajectory_id.tolist(); val_ids=val_lab.trajectory_id.tolist()
          Xtr=feat.loc[train_ids,cols].astype(float); Xval=feat.loc[val_ids,cols].astype(float)
          ytr=train_lab.ipi_affected.astype(int).to_numpy(); yval=val_lab.ipi_affected.astype(int).to_numpy()
          gtr=make_groups(train_lab).to_numpy()
          search,vs,thr=train_one(Xtr,ytr,gtr,Xval,yval,algo)
          eid=experiment_id(version,regime,algo,"SINGLE",source)
          model_path=MODELS_DIR/f"{eid}.joblib"; joblib.dump(search.best_estimator_,model_path)
          pd.DataFrame({"trajectory_id":val_ids,"y_true":yval,"score":vs,"threshold":thr}).to_csv(DEV_DIR/f"{eid}_validation_predictions.csv",index=False)
          m=metrics(yval,vs,thr)
          rows.append({"experiment_id":eid,"feature_version":version,"feature_regime":regime,"algorithm":algo,
             "strategy":"single_source","source_models":source,"target_model":source,"split":"validation",**m})
          search_rows.append({"experiment_id":eid,"best_cv_average_precision":search.best_score_,
                              "best_params":json.dumps(search.best_params_,sort_keys=True),
                              "model_sha256":sha256(model_path)})
          # Validation permutation importance, never Test.
          pi=permutation_importance(search.best_estimator_,Xval,yval,n_repeats=10,
                                    random_state=CFG["seed"],scoring="average_precision")
          for c,mean,std in zip(cols,pi.importances_mean,pi.importances_std):
            importance_rows.append({"experiment_id":eid,"feature":c,"importance_mean":mean,"importance_std":std})
        # Pooled leave-one-LLM-out
        for target in CFG["models"]:
          sources=[m for m in CFG["models"] if m!=target]
          tr=lab[(lab.target_model.isin(sources))&(lab.split=="train")].copy()
          va=lab[(lab.target_model.isin(sources))&(lab.split=="validation")].copy()
          # Assert disjointness globally for both group dimensions.
          assert_group_disjoint(tr,va)
          Xtr=feat.loc[tr.trajectory_id.tolist(),cols].astype(float)
          Xval=feat.loc[va.trajectory_id.tolist(),cols].astype(float)
          ytr=tr.ipi_affected.astype(int).to_numpy(); yval=va.ipi_affected.astype(int).to_numpy()
          gtr=make_groups(tr).to_numpy()
          search,vs,thr=train_one(Xtr,ytr,gtr,Xval,yval,algo)
          eid=experiment_id(version,regime,algo,"POOLED",f"LOLO_{target}")
          model_path=MODELS_DIR/f"{eid}.joblib"; joblib.dump(search.best_estimator_,model_path)
          pd.DataFrame({"trajectory_id":va.trajectory_id.tolist(),"y_true":yval,"score":vs,"threshold":thr}).to_csv(DEV_DIR/f"{eid}_validation_predictions.csv",index=False)
          m=metrics(yval,vs,thr)
          rows.append({"experiment_id":eid,"feature_version":version,"feature_regime":regime,"algorithm":algo,
             "strategy":"pooled_leave_one_out","source_models":"+".join(sources),"target_model":"SOURCE_VALIDATION_POOL","split":"validation",**m})
          search_rows.append({"experiment_id":eid,"best_cv_average_precision":search.best_score_,
                              "best_params":json.dumps(search.best_params_,sort_keys=True),
                              "model_sha256":sha256(model_path)})
          pi=permutation_importance(search.best_estimator_,Xval,yval,n_repeats=10,
                                    random_state=CFG["seed"],scoring="average_precision")
          for c,mean,std in zip(cols,pi.importances_mean,pi.importances_std):
            importance_rows.append({"experiment_id":eid,"feature":c,"importance_mean":mean,"importance_std":std})
    pd.DataFrame(rows).to_csv(DEV_DIR/"validation_metrics_all.csv",index=False)
    pd.DataFrame(search_rows).to_csv(DEV_DIR/"hyperparameter_search_summary.csv",index=False)
    pd.DataFrame(importance_rows).to_csv(DEV_DIR/"permutation_importance_validation.csv",index=False)
    make_feature_diagnostics(f77,f69,lab)
    return len(rows)

def make_feature_diagnostics(f77,f69,lab):
    # Train-only correlations and feature distribution KS comparisons.
    allcorr=[]
    for version,df in [("V77",f77),("V69",f69)]:
        ids=set(lab.loc[lab.split=="train","trajectory_id"])
        train=df[df.trajectory_id.isin(ids)].drop(columns="trajectory_id")
        corr=train.corr(numeric_only=True)
        corr.to_csv(DEV_DIR/f"{version}_train_feature_correlations.csv")
        for i,c1 in enumerate(corr.columns):
            for c2 in corr.columns[i+1:]:
                v=float(corr.loc[c1,c2])
                if abs(v)>=0.90: allcorr.append({"version":version,"feature_1":c1,"feature_2":c2,"pearson_r":v})
        merged=df.merge(lab[["trajectory_id","ipi_affected","split"]],on="trajectory_id")
        tr=merged[merged.split=="train"]
        for c in train.columns:
            x=tr.loc[tr.ipi_affected==1,c].dropna()
            y=tr.loc[tr.ipi_affected==0,c].dropna()
            if len(x)>0 and len(y)>0:
                ks=ks_2samp(x,y)
                pd.DataFrame([{"version":version,"feature":c,"ks_statistic":ks.statistic,"p_value":ks.pvalue,
                               "n_positive":len(x),"n_negative":len(y)}]).to_csv(
                    DEV_DIR/f"{version}_ks_train.csv",mode="a",header=not (DEV_DIR/f"{version}_ks_train.csv").exists(),index=False)
    pd.DataFrame(allcorr).to_csv(DEV_DIR/"high_correlation_pairs_train.csv",index=False)

def create_locked_predictions():
    f77,f69,meta,lab=load_data()
    feature_files={"V77":f77,"V69":f69}
    score_frames=[]
    for version,df in feature_files.items():
      df=df.set_index("trajectory_id")
      test_meta=meta[meta.split=="test"].copy()
      for regime in CFG["feature_regimes"]:
       cols=get_feature_cols(df.reset_index(),version,regime)
       for algo in CFG["algorithms"]:
        # Single source: same frozen threshold from source's validation-trained detector.
        for source in CFG["models"]:
          eid=experiment_id(version,regime,algo,"SINGLE",source)
          model=joblib.load(MODELS_DIR/f"{eid}.joblib")
          valpred=pd.read_csv(DEV_DIR/f"{eid}_validation_predictions.csv")
          thr=float(valpred.threshold.iloc[0])
          for target in CFG["models"]:
            tm=test_meta[test_meta.target_model==target]
            X=df.loc[tm.trajectory_id.tolist(),cols].astype(float)
            scores=model.predict_proba(X)[:,1]
            score_frames.append(pd.DataFrame({"experiment_id":eid,"training_strategy":"single_source",
              "source_model":source,"target_model":target,"feature_version":version,"feature_regime":regime,
              "algorithm":algo,"trajectory_id":tm.trajectory_id.tolist(),"score":scores,
              "threshold":thr,"predicted_label":(scores>=thr).astype(int)}))
        # Pooled: model trained on other two sources, threshold from source validation pool.
        for target in CFG["models"]:
          eid=experiment_id(version,regime,algo,"POOLED",f"LOLO_{target}")
          model=joblib.load(MODELS_DIR/f"{eid}.joblib")
          valpred=pd.read_csv(DEV_DIR/f"{eid}_validation_predictions.csv")
          thr=float(valpred.threshold.iloc[0])
          tm=test_meta[test_meta.target_model==target]
          X=df.loc[tm.trajectory_id.tolist(),cols].astype(float)
          scores=model.predict_proba(X)[:,1]
          score_frames.append(pd.DataFrame({"experiment_id":eid,"training_strategy":"pooled_leave_one_out",
            "source_model":"+".join(m for m in CFG["models"] if m!=target),"target_model":target,
            "feature_version":version,"feature_regime":regime,"algorithm":algo,
            "trajectory_id":tm.trajectory_id.tolist(),"score":scores,"threshold":thr,
            "predicted_label":(scores>=thr).astype(int)}))
    scores=pd.concat(score_frames,ignore_index=True)
    expected=4*3*3*3*78 + 4*3*3*78 # single 4 configs*3 algos*3 source*3 target*78 + pooled 4*3*3 target*78
    # Every single source-target cell has 78 target rows; pooled adds one 78-row target per config/algorithm.
    scores.to_csv(LOCK_DIR/"test_scores_locked.csv",index=False)
    score_hash=sha256(LOCK_DIR/"test_scores_locked.csv")
    lock={
      "status":"LOCKED_BEFORE_TEST_UNSEAL",
      "protocol":"P4_MASTER_EXPERIMENTAL_PROTOCOL_v1.0",
      "test_scores_file":"test_scores_locked.csv",
      "test_scores_sha256":score_hash,
      "row_count":int(len(scores)),
      "experiment_count":int(scores.experiment_id.nunique()),
      "config_sha256":sha256(ROOT/"config/experiment_config.json"),
      "v_exp_manifest_sha256":sha256(ROOT/"config/V_exp_manifest.json"),
      "models":{p.name:sha256(p) for p in sorted(MODELS_DIR.glob("*.joblib"))},
      "software":{"python":sys.version,"platform":platform.platform(),"sklearn":sklearn.__version__},
      "created_unix":time.time(),
      "test_labels_opened":False
    }
    (LOCK_DIR/"lock_manifest.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    return len(scores),score_hash

def evaluate_after_unseal(labels_path, outdir):
    # Explicitly separate command; caller must provide the externally stored sealed-label CSV after lock.
    lockpath=LOCK_DIR/"lock_manifest.json"
    if not lockpath.exists(): raise FileNotFoundError("No lock manifest. Refusing to evaluate.")
    lock=json.loads(lockpath.read_text())
    if lock.get("status")!="LOCKED_BEFORE_TEST_UNSEAL": raise ValueError("Scores are not locked.")
    if sha256(LOCK_DIR/"test_scores_locked.csv")!=lock["test_scores_sha256"]:
        raise ValueError("Locked score hash mismatch. Refusing evaluation.")
    labels=pd.read_csv(labels_path)
    required={"trajectory_id","ipi_affected"}
    if not required.issubset(labels.columns): raise ValueError("Sealed labels need trajectory_id,ipi_affected.")
    labels["trajectory_id"]=labels.trajectory_id.astype(str)
    if labels.trajectory_id.duplicated().any(): raise ValueError("Duplicate test labels.")
    if len(labels)!=234: raise ValueError("Expected exactly 234 sealed test labels.")
    scores=pd.read_csv(LOCK_DIR/"test_scores_locked.csv")
    merged=scores.merge(labels[["trajectory_id","ipi_affected"]],on="trajectory_id",how="left",validate="many_to_one")
    if merged.ipi_affected.isna().any(): raise ValueError("Some score rows have no test label.")
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    groupcols=["experiment_id","training_strategy","source_model","target_model","feature_version","feature_regime","algorithm"]
    for keys,g in merged.groupby(groupcols):
        m=metrics(g.ipi_affected.astype(int),g.score,g.threshold.iloc[0])
        rows.append(dict(zip(groupcols,keys),**m))
    pd.DataFrame(rows).to_csv(out/"test_metrics_by_cell.csv",index=False)
    merged.to_csv(out/"test_predictions_with_labels_after_unseal.csv",index=False)
    # Group-aware bootstrap uses condition/user-task group identifiers from frozen metadata.
    meta=pd.read_csv(DATA/"metadata_membership_1131.csv",dtype={"trajectory_id":str})
    gmeta=meta[["trajectory_id","condition_group_id","user_task_group_id"]]
    merged=merged.merge(gmeta,on="trajectory_id",how="left",validate="many_to_one")
    rng=np.random.default_rng(CFG["seed"])
    bootrows=[]
    for keys,g in merged.groupby(groupcols):
        # Resample user-task groups as primary independent units.
        unit=g.user_task_group_id.fillna(g.condition_group_id).astype(str)
        units=unit.unique()
        vals=[]
        for _ in range(CFG["bootstrap_replicates"]):
            sampled=rng.choice(units,size=len(units),replace=True)
            idx=np.concatenate([np.where(unit.to_numpy()==u)[0] for u in sampled])
            b=g.iloc[idx]
            if b.ipi_affected.nunique()<2: continue
            vals.append([roc_auc_score(b.ipi_affected,b.score),
                         average_precision_score(b.ipi_affected,b.score),
                         balanced_accuracy_score(b.ipi_affected,(b.score>=b.threshold).astype(int))])
        if vals:
            arr=np.asarray(vals)
            for j,metric_name in enumerate(["auroc","average_precision","balanced_accuracy"]):
                bootrows.append(dict(zip(groupcols,keys),metric=metric_name,
                    ci_low=float(np.quantile(arr[:,j],.025)),ci_high=float(np.quantile(arr[:,j],.975)),
                    valid_replicates=len(vals)))
    pd.DataFrame(bootrows).to_csv(out/"group_bootstrap_95ci.csv",index=False)
    return len(rows)

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=["train","lock","all-dev","evaluate"])
    ap.add_argument("--labels",help="Path to externally stored sealed test-label CSV; only after score lock.")
    ap.add_argument("--out",default=str(OUT/"final_evaluation"))
    args=ap.parse_args()
    if args.command in ("train","all-dev"):
        n=save_dev_artifacts(); print(f"Completed development models: {n}")
    if args.command in ("lock","all-dev"):
        n,h=create_locked_predictions(); print(f"Locked test score rows: {n}; SHA256={h}")
    if args.command=="evaluate":
        if not args.labels: ap.error("--labels required after authorized unsealing.")
        n=evaluate_after_unseal(args.labels,args.out); print(f"Evaluated cells: {n}")

if __name__=="__main__": main()
