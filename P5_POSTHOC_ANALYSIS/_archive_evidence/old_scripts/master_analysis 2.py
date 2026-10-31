#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, platform, sys, traceback
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

MODELS=["LLM-A","LLM-B","LLM-C"]
METRICS=["auroc","average_precision","balanced_accuracy","f1","precision","recall_tpr"]

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def holm(pvals):
    p=np.asarray(pvals,dtype=float); n=len(p)
    order=np.argsort(p); adj=np.empty(n); running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(n-rank)*p[idx])
        adj[idx]=min(1.0,running)
    return adj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--p4",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); p4=Path(a.p4).resolve(); out=Path(a.out).resolve()
    start=datetime.now(timezone.utc).isoformat()
    statuses=[]; errors=[]
    def rec(name,status,detail):
        statuses.append({"module":name,"status":status,"detail":str(detail)})
        print(f"[{status}] {name}: {detail}",flush=True)
    def read(rel): return pd.read_csv(p4/rel)
    try:
        metrics=read("outputs/final_test_evaluation/test_metrics_by_cell.csv")
        fair=read("outputs/final_test_evaluation/fair_cross_llm_comparison.csv")
        boot=read("outputs/final_test_evaluation/group_bootstrap_95ci.csv")
        locked=read("outputs/locked_test_scores/test_scores_locked.csv")
        labels=read("outputs/after_unseal_inputs/test_labels_234.csv")
        membership=read("data/metadata_membership_1131.csv")
        preds=read("outputs/final_test_evaluation/test_predictions_with_labels_after_unseal.csv")
        permutation=read("outputs/development/permutation_importance_validation.csv")
        ks69=read("outputs/development/V69_ks_train.csv")
        ks77=read("outputs/development/V77_ks_train.csv")
        assert len(metrics)==144, f"Expected 144 metrics rows, got {len(metrics)}"
        assert len(locked)==11232, f"Expected 11232 locked rows, got {len(locked)}"
        assert len(labels)==234, f"Expected 234 test labels, got {len(labels)}"
        rec("Input and schema audit","PASS",f"metrics={len(metrics)}, locked={len(locked)}, labels={len(labels)}")
    except Exception as e:
        rec("Input and schema audit","FAILED",repr(e)); errors.append(traceback.format_exc())
        metrics=None

    if metrics is not None:
      try:
        single=metrics[metrics.training_strategy.eq("single_source")].copy()
        keys=["feature_version","feature_regime","algorithm"]; rows=[]
        for key,g in single.groupby(keys):
          for s in MODELS:
            d=g[(g.source_model==s)&(g.target_model==s)]
            if len(d)!=1: continue
            for t in MODELS:
              if t==s: continue
              c=g[(g.source_model==s)&(g.target_model==t)]
              td=g[(g.source_model==t)&(g.target_model==t)]
              if len(c)!=1: continue
              for m in METRICS:
                if m not in g: continue
                sv=float(d.iloc[0][m]); cv=float(c.iloc[0][m])
                tv=float(td.iloc[0][m]) if len(td)==1 else np.nan
                rows.append(dict(zip(keys,key))|{"source_model":s,"target_model":t,
                  "metric":m,"source_in_domain":sv,"cross_model":cv,
                  "gap_src":sv-cv,"target_in_domain":tv,
                  "gap_tgt":tv-cv if np.isfinite(tv) else np.nan})
        gaps=pd.DataFrame(rows)
        gaps.to_csv(out/"01_generalization_gap/results/formal_single_source_gaps.csv",index=False)
        summ=(gaps.groupby(keys+["metric"]).agg(
          mean_gap_src=("gap_src","mean"),sd_gap_src=("gap_src","std"),
          mean_gap_tgt=("gap_tgt","mean"),n_pairs=("gap_src","count")).reset_index())
        summ.to_csv(out/"01_generalization_gap/results/generalization_gap_summary.csv",index=False)
        rec("Formal single-source generalization gaps","PASS",f"{len(gaps)} directed comparisons")
      except Exception as e:
        rec("Formal single-source generalization gaps","FAILED",repr(e)); errors.append(traceback.format_exc())

      try:
        pooled=metrics[metrics.training_strategy.eq("pooled_leave_one_out")]
        cross=single[single.source_model.ne(single.target_model)]
        group=["target_model","feature_version","feature_regime","algorithm"]; rows=[]
        for key,pg in pooled.groupby(group):
          if len(pg)!=1: continue
          cg=cross
          for col,val in zip(group,key): cg=cg[cg[col].eq(val)]
          if len(cg)!=2: continue
          r=dict(zip(group,key))
          for m in METRICS:
            if m in metrics:
              r[f"pooled_{m}"]=float(pg.iloc[0][m])
              r[f"single_cross_mean_{m}"]=float(cg[m].mean())
              r[f"difference_{m}"]=r[f"pooled_{m}"]-r[f"single_cross_mean_{m}"]
          rows.append(r)
        pd.DataFrame(rows).to_csv(out/"01_generalization_gap/results/pooled_lolo_vs_single_cross.csv",index=False)
        rec("Pooled LOLO comparison","PASS",f"{len(rows)} cells; kept separate from formal single-source gap")
      except Exception as e:
        rec("Pooled LOLO comparison","FAILED",repr(e)); errors.append(traceback.format_exc())

      try:
        # Paired unit = target x feature version x regime x algorithm.
        # Compare pooled LOLO with mean of the two single-source off-diagonal models.
        pooled=metrics[metrics.training_strategy.eq("pooled_leave_one_out")]
        cross=single[single.source_model.ne(single.target_model)]
        pairkeys=["target_model","feature_version","feature_regime","algorithm"]
        result=[]
        for m in METRICS:
          if m not in metrics: continue
          aa=pooled.groupby(pairkeys)[m].mean().rename("pooled")
          bb=cross.groupby(pairkeys)[m].mean().rename("single_cross")
          z=pd.concat([aa,bb],axis=1).dropna()
          if len(z)<2: continue
          diff=z.pooled-z.single_cross
          wp=wilcoxon(diff).pvalue if np.any(diff!=0) else np.nan
          tp=ttest_rel(z.pooled,z.single_cross).pvalue
          result.append({"metric":m,"n_matched_units":len(z),
            "mean_pooled":z.pooled.mean(),"mean_single_cross":z.single_cross.mean(),
            "mean_paired_difference":diff.mean(),"median_paired_difference":diff.median(),
            "wilcoxon_p":wp,"paired_t_p":tp})
        st=pd.DataFrame(result)
        for col in ["wilcoxon_p","paired_t_p"]:
          valid=st[col].notna()
          st[col+"_holm"]=np.nan
          if valid.any(): st.loc[valid,col+"_holm"]=holm(st.loc[valid,col].values)
        st.to_csv(out/"reports/tables/pooled_vs_single_paired_statistics.csv",index=False)
        rec("Matched pooled-vs-single statistics","PASS",f"{len(st)} metric tests; exploratory matched summaries")
      except Exception as e:
        rec("Matched pooled-vs-single statistics","FAILED",repr(e)); errors.append(traceback.format_exc())

      try:
        boot.to_csv(out/"03_distribution_shift/results/group_bootstrap_ci_review.csv",index=False)
        ks69.to_csv(out/"03_distribution_shift/results/V69_KS_existing.csv",index=False)
        ks77.to_csv(out/"03_distribution_shift/results/V77_KS_existing.csv",index=False)
        permutation.to_csv(out/"02_feature_importance/results/permutation_importance_validation_existing.csv",index=False)
        fair.to_csv(out/"reports/tables/fair_cross_llm_comparison.csv",index=False)
        rec("Existing P4 analysis review","PASS","Copied bootstrap CI, KS, validation permutation and fair comparison tables")
      except Exception as e:
        rec("Existing P4 analysis review","FAILED",repr(e)); errors.append(traceback.format_exc())

      try:
        pd.DataFrame([{
          "metric_rows":len(metrics),"locked_score_rows":len(locked),
          "prediction_rows":len(preds),"test_label_rows":len(labels),
          "test_membership_rows":int(membership.split.astype(str).str.lower().eq("test").sum()),
          "unique_test_ids":labels.trajectory_id.nunique()
        }]).to_csv(out/"00_compliance/input_count_audit.csv",index=False)
        rec("Frozen artifact count audit","PASS","Counts recorded; source P4 files not modified")
      except Exception as e:
        rec("Frozen artifact count audit","FAILED",repr(e)); errors.append(traceback.format_exc())

    # These require additional prespecified methods or data. Mark honestly.
    rec("SHAP","BLOCKED","Requires prespecified background/explanation cohort and estimator-specific validation.")
    rec("Feature ablation","BLOCKED","Requires a prespecified ablation design and secondary model training; P4 remains frozen.")
    rec("Random/rule/text baselines","BLOCKED","Baseline definitions/implementations are not present in supplied P4 artifacts.")
    rec("Label sensitivity","BLOCKED","Requires a prespecified sensitivity protocol; frozen labels are not modified.")
    rec("Pbase/payload shift","BLOCKED","Required comparison dataset and mapping are not present in supplied schema.")

    # Verify original snapshot manifest if available; expected format: path, sha256 columns.
    try:
      snap=out/"provenance/source_hashes.csv"
      if snap.exists():
        sdf=pd.read_csv(snap); cols={c.lower():c for c in sdf.columns}
        pc=cols.get("path") or sdf.columns[0]; hc=cols.get("sha256") or sdf.columns[1]
        missing=[]; mismatch=[]
        for _,r in sdf.iterrows():
          f=p4/str(r[pc])
          if not f.exists(): missing.append(str(r[pc]))
          elif sha256(f)!=str(r[hc]): mismatch.append(str(r[pc]))
        rec("P4 snapshot verification","PASS" if not missing and not mismatch else "FAILED",
            f"checked={len(sdf)}, missing={len(missing)}, mismatches={len(mismatch)}")
      else: rec("P4 snapshot verification","REVIEW","source_hashes.csv not found at P5/provenance")
    except Exception as e:
      rec("P4 snapshot verification","FAILED",repr(e)); errors.append(traceback.format_exc())

    sdf=pd.DataFrame(statuses)
    sdf.to_csv(out/"reports/tables/module_status.csv",index=False)
    end=datetime.now(timezone.utc).isoformat()
    limitations=[
      "Pooled LOLO is reported separately from the paper's single-source gap definitions.",
      "Matched tests compare target/configuration/algorithm summaries; they are exploratory and not independent-cell inference.",
      "P4 permutation importance is validation-based, not a test-level importance estimate.",
      "P4 validation thresholds were selected on validation by maximum balanced accuracy; validation metrics are tuned.",
      "P4 bootstrap resamples user_task_group_id with condition_group_id fallback, not necessarily connected components.",
      "SHAP, ablation, baselines, label sensitivity and Pbase/payload shift are BLOCKED pending a prespecified protocol/data."
    ]
    manifest={"started_utc":start,"finished_utc":end,"python":sys.version,
      "platform":platform.platform(),"p4_path":str(p4),"p5_path":str(out),
      "p4_read_only":True,"module_status":statuses,"errors":errors,
      "limitations":limitations}
    (out/"provenance/p5_run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str))
    report=["# P5 Master Post-hoc Analysis Report","",f"Started UTC: {start}",f"Finished UTC: {end}",
      "P4 read-only: YES","","## Module status","",sdf.to_markdown(index=False),"",
      "## Interpretation limitations",""]+[f"- {x}" for x in limitations]+[
      "","## Main outputs",
      "- `01_generalization_gap/results/formal_single_source_gaps.csv`",
      "- `01_generalization_gap/results/generalization_gap_summary.csv`",
      "- `01_generalization_gap/results/pooled_lolo_vs_single_cross.csv`",
      "- `reports/tables/pooled_vs_single_paired_statistics.csv`",
      "- `03_distribution_shift/results/group_bootstrap_ci_review.csv`",
      "- `03_distribution_shift/results/V69_KS_existing.csv`",
      "- `03_distribution_shift/results/V77_KS_existing.csv`",
      "- `02_feature_importance/results/permutation_importance_validation_existing.csv`",
      "- `reports/tables/module_status.csv`",
      "- `provenance/p5_run_manifest.json`","",
      "Blocked analyses are reported as blocked; no results are fabricated."]
    (out/"reports/P5_MASTER_REPORT.md").write_text("\n".join(report))
    print("\nREPORT:",out/"reports/P5_MASTER_REPORT.md")
    print("STATUS:",sdf.status.value_counts().to_dict())
    return 1 if (sdf.status=="FAILED").any() else 0

if __name__=="__main__": raise SystemExit(main())
