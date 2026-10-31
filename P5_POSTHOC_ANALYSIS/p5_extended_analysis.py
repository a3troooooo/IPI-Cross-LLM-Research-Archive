#!/usr/bin/env python3
"""
P5 Extended Analysis: SHAP, feature ablation, grouped inference, baselines.
P4 is treated as read-only. All new artifacts are written under P5.
Exploratory post-unseal test analyses are explicitly labelled exploratory.
"""
from __future__ import annotations
import argparse, json, hashlib, traceback, sys, platform, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from scipy.stats import wilcoxon
from sklearn.metrics import (roc_auc_score, average_precision_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score)
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

SEED=20261031
MODELS=["LLM-A","LLM-B","LLM-C"]
METRICS=["auroc","average_precision","balanced_accuracy","precision","recall","f1"]

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def metric_row(y, score, threshold=.5):
    y=np.asarray(y,dtype=int); score=np.asarray(score,dtype=float)
    pred=(score>=threshold).astype(int)
    return {
      "n":len(y),"positive":int(y.sum()),"prevalence":float(y.mean()),
      "auroc":float(roc_auc_score(y,score)) if len(np.unique(y))==2 else np.nan,
      "average_precision":float(average_precision_score(y,score)) if len(np.unique(y))==2 else np.nan,
      "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
      "precision":float(precision_score(y,pred,zero_division=0)),
      "recall":float(recall_score(y,pred,zero_division=0)),
      "f1":float(f1_score(y,pred,zero_division=0))
    }

def groups_union_find(frame):
    cols=["condition_group_id","user_task_group_id"]
    parent=list(range(len(frame))); rank=[0]*len(frame); seen={}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a=find(a); b=find(b)
        if a==b:return
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    for i,row in enumerate(frame[cols].itertuples(index=False,name=None)):
        for prefix,val in zip(("C","U"),row):
            if pd.isna(val):continue
            key=(prefix,str(val))
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    return np.asarray([find(i) for i in range(len(frame))])

def get_model_file(p4, version, algo, strategy, source):
    # P4 model filenames use SINGLE and POOLED_LOLO.
    suffix=f"P4_{version}_V_all_{algo}_{strategy}_{source}_v1.joblib"
    return p4/"outputs/models"/suffix

def feature_groups(cols):
    # Semantic groups based on P4's feature names; groups are exhaustive and disjoint.
    rules={
      "tool_activity":lambda c:c in {"n_tool_calls","n_tool_results","n_unique_tools","n_unmapped_tool_calls","trajectory_length","max_calls_per_step","mean_calls_per_step"},
      "action_semantics":lambda c:c.startswith("n_act_") or c in {"n_read","n_write","n_egress","n_destructive","frac_write","frac_egress","has_write"},
      "action_sequence":lambda c:c.startswith("trans_") or c in {"first_write_pos","n_reads_after_first_write","n_write_bursts","max_read_streak","action_entropy","n_tool_revisits","max_same_tool_run","n_repeated_call_shapes","n_domains","n_domain_switches","n_write_domains","n_writes_new_domain"},
      "exposure_structure":lambda c:c in {"n_freetext_reads","n_writes_after_freetext","n_egress_after_freetext","n_destructive_after_freetext","frac_writes_after_freetext","n_calls_after_last_freetext"},
      "tool_errors":lambda c:c.startswith("n_error_") or c in {"n_tool_errors","frac_tool_errors","n_retry_after_error"},
      "argument_structure":lambda c:c.startswith("n_calls_") or c.startswith("frac_arg_") or c in {"mean_args_per_call","max_args_per_call"},
      "conversation_structure":lambda c:c in {"ended_with_final","final_output_present","n_steps","n_agent_messages","n_steps_message_and_call"}
    }
    out={k:[c for c in cols if fn(c)] for k,fn in rules.items()}
    assigned=set(sum(out.values(),[]))
    out["other"]=[c for c in cols if c not in assigned]
    return {k:v for k,v in out.items() if v}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--p4",required=True); ap.add_argument("--p5",required=True)
    a=ap.parse_args(); p4=Path(a.p4); p5=Path(a.p5)
    start=datetime.now(timezone.utc).isoformat()
    status=[]; errors=[]
    def log(mod,st,msg):
        status.append({"module":mod,"status":st,"detail":str(msg)})
        print(f"[{st}] {mod}: {msg}",flush=True)
    # required inputs
    files={
      "f77":p4/"data/features_p3_77_1131.csv",
      "f69":p4/"data/features_p4_69_1131.csv",
      "meta":p4/"data/metadata_membership_1131.csv",
      "trainlabels":p4/"data/train_validation_labels_897.csv",
      "testlabels":p4/"outputs/after_unseal_inputs/test_labels_234.csv",
      "preds":p4/"outputs/final_test_evaluation/test_predictions_with_labels_after_unseal.csv",
      "metrics":p4/"outputs/final_test_evaluation/test_metrics_by_cell.csv",
      "hyper":p4/"outputs/development/hyperparameter_search_summary.csv"
    }
    try:
      missing=[str(v) for v in files.values() if not v.exists()]
      if missing: raise FileNotFoundError("Missing P4 inputs: "+", ".join(missing))
      f77=pd.read_csv(files["f77"]); f69=pd.read_csv(files["f69"])
      meta=pd.read_csv(files["meta"]); trlab=pd.read_csv(files["trainlabels"])
      telab=pd.read_csv(files["testlabels"]); preds=pd.read_csv(files["preds"])
      metrics=pd.read_csv(files["metrics"]); hyper=pd.read_csv(files["hyper"])
      for df in [f77,f69,meta,trlab,telab,preds]:
        if "trajectory_id" in df: df["trajectory_id"]=df.trajectory_id.astype(str)
      log("Input audit","PASS",f"V77={f77.shape}; V69={f69.shape}; train/val labels={len(trlab)}; test labels={len(telab)}")
    except Exception as e:
      log("Input audit","FAILED",repr(e)); (p5/"reports").mkdir(exist_ok=True)
      (p5/"reports/P5_EXTENDED_REPORT.md").write_text("# P5 Extended Analysis\n\nInput audit failed:\n\n"+repr(e))
      return 2

    # Connected-component group audit; ensure split disjointness in components.
    try:
      mm=meta.copy(); mm["component"]=groups_union_find(mm)
      split_comps=mm.groupby("component").split.nunique()
      leakage=int((split_comps>1).sum())
      pd.DataFrame({"component":mm.component,"trajectory_id":mm.trajectory_id,"split":mm.split}).to_csv(
        p5/"10_STATISTICAL_INFERENCE/results/component_membership.csv",index=False)
      log("Connected-component audit","PASS" if leakage==0 else "REVIEW",
          f"{mm.component.nunique()} components; components spanning splits={leakage}")
    except Exception as e:
      log("Connected-component audit","FAILED",repr(e)); errors.append(traceback.format_exc()); mm=None

    # SHAP: representative V77/V_all model set (3 algos x 3 single-source + 3 pooled).
    try:
      import shap
      Xall=f77.set_index("trajectory_id").drop(columns=[],errors="ignore")
      feature_cols=[c for c in Xall.columns if c!="trajectory_id"]
      train_only = trlab[trlab["split"].astype(str).str.lower().eq("train")].copy()
      ytrain=train_only.set_index("trajectory_id")
      Xtrain=Xall.loc[ytrain.index,feature_cols].astype(float)
      shap_rows=[]; direction_rows=[]; successes=0
      shap_dir=p5/"08_SHAP/figures"; shap_dir.mkdir(parents=True,exist_ok=True)
      for algo in ["LR","RF","GB"]:
        configs=[("SINGLE",m,m) for m in MODELS]+[("POOLED_LOLO",m,m) for m in MODELS]
        for strategy,source,target in configs:
          model_path=get_model_file(p4,"V77",algo,strategy,source)
          if not model_path.exists(): continue
          model=joblib.load(model_path)
          # Explain target-specific test records; background is Train only.
          target_ids=telab.loc[telab.target_model.eq(target),"trajectory_id"].astype(str).tolist()
          if not target_ids: continue
          Xbg=Xtrain.sample(n=min(100,len(Xtrain)),random_state=SEED)
          Xexp=Xall.loc[target_ids,feature_cols].astype(float).head(78)
          # Explain positive-class probability. TreeExplainer on transformed features avoids
          # pipeline incompatibilities; LR uses LinearExplainer on transformed features.
          imp=model.named_steps["imputer"]
          Zbg=imp.transform(Xbg)
          Zexp=imp.transform(Xexp)
          est=model.named_steps["model"]
          if algo=="LR":
            expl=shap.LinearExplainer(est, Zbg)
            sv=expl.shap_values(Zexp)
          else:
            expl=shap.TreeExplainer(
                est,
                data=Zbg,
                feature_perturbation="interventional"
            )
            sv=expl.shap_values(Zexp)
          if isinstance(sv,list): sv=sv[1]
          sv=np.asarray(sv)
          if sv.ndim==3: sv=sv[:,:,1]
          if sv.shape[1]!=len(feature_cols): raise ValueError(f"Unexpected SHAP shape {sv.shape}")
          mean_abs=np.mean(np.abs(sv),axis=0)
          mean_signed=np.mean(sv,axis=0)
          for j,c in enumerate(feature_cols):
            shap_rows.append({"algorithm":algo,"strategy":strategy,"source":source,"target":target,
              "feature":c,"mean_abs_shap":float(mean_abs[j]),"mean_signed_shap":float(mean_signed[j]),
              "n_explained":len(Xexp),"background":"train_only_sample_100"})
            direction_rows.append({"algorithm":algo,"strategy":strategy,"source":source,"target":target,
              "feature":c,"direction":"positive" if mean_signed[j]>0 else "negative" if mean_signed[j]<0 else "neutral",
              "mean_signed_shap":float(mean_signed[j])})
          successes+=1
      pd.DataFrame(shap_rows).to_csv(p5/"08_SHAP/results/shap_global_importance.csv",index=False)
      pd.DataFrame(direction_rows).to_csv(p5/"08_SHAP/results/shap_direction_by_model.csv",index=False)
      if len(shap_rows):
        cons=(pd.DataFrame(shap_rows).groupby(["algorithm","strategy","feature"])
              .agg(mean_abs_shap=("mean_abs_shap","mean"),mean_signed_shap=("mean_signed_shap","mean"),
                   n_model_contexts=("target","count")).reset_index())
        cons.to_csv(p5/"08_SHAP/results/shap_cross_llm_consistency.csv",index=False)
      log("SHAP","PASS" if successes else "BLOCKED",f"{successes}/18 model contexts explained; V77/V_all; test explanation is post-unseal exploratory")
    except Exception as e:
      log("SHAP","FAILED",repr(e)); errors.append(traceback.format_exc())

    # Grouped inference over P4 locked predictions. Resample connected components.
    try:
      joined=preds.merge(telab[["trajectory_id","target_model","ipi_affected"]].drop_duplicates(),
                         on=["trajectory_id","target_model"],how="inner",suffixes=("","_label"))
      joined["component"]=joined.trajectory_id.map(mm.set_index("trajectory_id").component) if mm is not None else joined.trajectory_id
      rng=np.random.default_rng(SEED); bootrows=[]
      # For each experiment and metric, resample whole connected components.
      for eid,g in joined.groupby("experiment_id"):
        comps=g.component.unique()
        if len(comps)<2: continue
        for metric in ["auroc","average_precision"]:
          vals=[]
          for _ in range(1000):
            sampled=rng.choice(comps,size=len(comps),replace=True)
            chunks=[g[g.component.eq(c)] for c in sampled]
            b=pd.concat(chunks,ignore_index=True)
            if b.ipi_affected.nunique()<2: continue
            try:
              vals.append(roc_auc_score(b.ipi_affected,b.score) if metric=="auroc"
                          else average_precision_score(b.ipi_affected,b.score))
            except Exception: pass
          if vals:
            point=roc_auc_score(g.ipi_affected,g.score) if metric=="auroc" else average_precision_score(g.ipi_affected,g.score)
            bootrows.append({"experiment_id":eid,"metric":metric,"estimate":point,
              "ci_low":np.quantile(vals,.025),"ci_high":np.quantile(vals,.975),
              "valid_replicates":len(vals),"bootstrap_replicates":1000,"resampling_unit":"connected_component"})
      pd.DataFrame(bootrows).to_csv(p5/"10_STATISTICAL_INFERENCE/results/connected_component_bootstrap_ci.csv",index=False)
      log("Connected-component bootstrap","PASS",f"{len(bootrows)} experiment-metric intervals")
    except Exception as e:
      log("Connected-component bootstrap","FAILED",repr(e)); errors.append(traceback.format_exc())

    # Baselines: fixed rule, random scores, prevalence constant; all on test labels,
    # explicitly exploratory after unseal. Rule uses only P4 predeclared exposure/action fields.
    try:
      feat=f77.set_index("trajectory_id")
      lab=telab.set_index("trajectory_id")
      b_rows=[]
      for target in MODELS:
        ids=lab[lab.target_model.eq(target)].index.astype(str)
        y=lab.loc[ids,"ipi_affected"].astype(int).to_numpy()
        X=feat.loc[ids]
        rule=((X["n_writes_after_freetext"].fillna(0)>0) |
              (X["n_egress_after_freetext"].fillna(0)>0) |
              (X["n_destructive_after_freetext"].fillna(0)>0)).astype(int).to_numpy()
        # A fixed binary rule has scores 0/1; no threshold fitted on test.
        for name,score in [("rule_exposure_then_sensitive_action",rule.astype(float))]:
          b_rows.append({"target_model":target,"baseline":name,**metric_row(y,score,.5),
                         "status":"exploratory_test_post_unseal"})
        rng=np.random.default_rng(SEED+MODELS.index(target))
        for rep in range(200):
          score=rng.random(len(y))
          b_rows.append({"target_model":target,"baseline":f"random_score_rep_{rep+1}",**metric_row(y,score,.5),
                         "status":"exploratory_test_post_unseal"})
        prevalence=float(y.mean())
        b_rows.append({"target_model":target,"baseline":"prevalence_constant",
          "n":len(y),"positive":int(y.sum()),"prevalence":prevalence,"auroc":np.nan,
          "average_precision":prevalence,"balanced_accuracy":.5,"precision":np.nan,
          "recall":0.0,"f1":0.0,"status":"no_ranking_information"})
      bdf=pd.DataFrame(b_rows); bdf.to_csv(p5/"11_BASELINE_COMPARISON/results/baseline_test_metrics.csv",index=False)
      summ=(bdf[bdf.baseline.str.startswith("random_score_rep")].groupby("target_model")
            .agg(random_auroc_mean=("auroc","mean"),random_auroc_sd=("auroc","std"),
                 random_ap_mean=("average_precision","mean"),random_ap_sd=("average_precision","std")).reset_index())
      summ.to_csv(p5/"11_BASELINE_COMPARISON/results/random_baseline_summary.csv",index=False)
      log("Random/prevalence/rule baselines","PASS",f"{len(bdf)} rows; rule fixed before test evaluation; exploratory")
    except Exception as e:
      log("Baseline comparison","FAILED",repr(e)); errors.append(traceback.format_exc())

    # Feature ablation is implemented as a protocol-gated module: use training data and
    # saved P4 hyperparameters, validation-selected thresholds; do not tune on Test.
    try:
      groups=feature_groups([c for c in f77.columns if c!="trajectory_id"])
      pd.DataFrame([{"feature_group":k,"n_features":len(v),"features":";".join(v)} for k,v in groups.items()]).to_csv(
        p5/"09_FEATURE_ABLATION/results/feature_group_manifest.csv",index=False)
      # This first executable package emits the complete pre-registered ablation matrix.
      # Actual refits are intentionally gated pending manual confirmation of group taxonomy.
      matrix=[]
      for strategy in ["single_source","pooled_leave_one_out"]:
        for model in MODELS:
          for group in groups:
            matrix.append({"strategy":strategy,"target_or_source":model,"feature_group_removed":group,
                           "status":"READY_FOR_SECONDARY_REFIT","feature_version":"V77","feature_regime":"V_all"})
      pd.DataFrame(matrix).to_csv(p5/"09_FEATURE_ABLATION/results/ablation_design_matrix.csv",index=False)
      log("Feature ablation design","BLOCKED","Feature groups and ablation matrix generated; refits require approval of semantic grouping before execution")
    except Exception as e:
      log("Feature ablation design","FAILED",repr(e)); errors.append(traceback.format_exc())

    # Report and manifest
    sdf=pd.DataFrame(status)
    sdf.to_csv(p5/"reports/P5_EXTENDED_MODULE_STATUS.csv",index=False)
    finish=datetime.now(timezone.utc).isoformat()
    limitations=[
      "SHAP explanations use a Train-only background but explain post-unseal Test predictions; exploratory, not confirmatory.",
      "SHAP run is limited to V77/V_all and 18 contexts (3 algorithms x 6 training contexts).",
      "Feature ablation refits are not run: the generated feature-group taxonomy must be approved before secondary refitting.",
      "Connected-component bootstrap resamples components; this is a new post-hoc interval analysis, not the original P4 bootstrap.",
      "Rule baseline is a fixed exposure-then-sensitive-action heuristic, not a validated domain rule; report as exploratory.",
      "Random baseline uses 200 Monte Carlo score draws per target; prevalence constant has no AUROC ranking information.",
      "No P4 models, labels, scores, thresholds, or manifests are modified."
    ]
    manifest={"started_utc":start,"finished_utc":finish,"python":sys.version,"platform":platform.platform(),
      "p4_path":str(p4),"p5_path":str(p5),"p4_read_only":True,"modules":status,
      "limitations":limitations,"errors":errors}
    (p5/"provenance").mkdir(exist_ok=True)
    (p5/"provenance/p5_extended_manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
    report=["# P5 Extended Analysis Report","",f"Started UTC: {start}",f"Finished UTC: {finish}",
      "**P4 read-only: YES**","","## Module status","",sdf.to_string(index=False),"",
      "## Interpretation limitations",""]+[f"- {x}" for x in limitations]+[
      "","## Outputs",
      "- `08_SHAP/results/shap_global_importance.csv`",
      "- `08_SHAP/results/shap_direction_by_model.csv`",
      "- `08_SHAP/results/shap_cross_llm_consistency.csv`",
      "- `09_FEATURE_ABLATION/results/feature_group_manifest.csv`",
      "- `09_FEATURE_ABLATION/results/ablation_design_matrix.csv`",
      "- `10_STATISTICAL_INFERENCE/results/connected_component_bootstrap_ci.csv`",
      "- `11_BASELINE_COMPARISON/results/baseline_test_metrics.csv`",
      "- `11_BASELINE_COMPARISON/results/random_baseline_summary.csv`",
      "- `reports/P5_EXTENDED_MODULE_STATUS.csv`",
      "- `provenance/p5_extended_manifest.json`","",
      "Do not describe blocked modules as completed. All Test-based post-unseal analyses are exploratory."]
    (p5/"reports/P5_EXTENDED_REPORT.md").write_text("\n".join(report))
    print("\nREPORT: reports/P5_EXTENDED_REPORT.md")
    print("STATUS:",sdf.status.value_counts().to_dict())
    return 1 if (sdf.status=="FAILED").any() else 0

if __name__=="__main__": raise SystemExit(main())
