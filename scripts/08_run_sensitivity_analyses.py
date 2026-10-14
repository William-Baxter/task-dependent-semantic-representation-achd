from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    BASE_NUMERIC, BASE_CATEGORICAL, CV_SEED, PEDSQL_CONTINUOUS, DASS_CONTINUOUS,
    PEDSQL_ORDINAL, DASS_ORDINAL, PRO_ORDINAL, add_lesion_family_11,
    bh_adjust, derive_binary_outcomes, normalise_ids, one_hot_encoder,
    ordinal_pro_frame, outcome_specs, read_json,
)

N_SPLITS=5
N_REPEATS=10
KMEANS_N_INIT=10
KMEANS_SEED=42


def load_data(path, labels_path):
    df=pd.read_csv(path,dtype={"record_id":"string"},low_memory=False); lab=pd.read_csv(labels_path,dtype={"record_id":"string"})
    df.record_id=normalise_ids(df.record_id); lab.record_id=normalise_ids(lab.record_id)
    overlap=[c for c in PRO_ORDINAL if c in df]
    if overlap: df=df.drop(columns=overlap)
    df=df.merge(lab,on="record_id",validate="one_to_one"); df=derive_binary_outcomes(df); df=add_lesion_family_11(df)
    return df.sort_values("record_id",kind="mergesort").reset_index(drop=True)


def model_pipeline(binary, partition=False, burden=False):
    nums=list(BASE_NUMERIC)+(["burden"] if burden else [])
    cats=list(BASE_CATEGORICAL)+(["partition"] if partition else [])
    prep=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",one_hot_encoder())]),cats),
    ])
    mdl=LogisticRegression(C=1e6,solver="liblinear",max_iter=5000,random_state=CV_SEED) if binary else LinearRegression()
    return Pipeline([("prep",prep),("model",mdl)])


def frame(df, burden=None, partition=None):
    x=df[BASE_NUMERIC+BASE_CATEGORICAL].copy()
    if burden is not None: x["burden"]=np.asarray(burden,float)
    if partition is not None: x["partition"]=pd.Series(partition,index=x.index).astype("string").to_numpy()
    return x


def pooled_metric(y,p,binary): return float(roc_auc_score(y,p) if binary else r2_score(y,p))


def median_split_benchmark(df, repeats):
    groups=df.lesion_family_11.astype(str).to_numpy(); rows=[]
    for repeat in range(1,repeats+1):
        splitter=StratifiedKFold(N_SPLITS,shuffle=True,random_state=CV_SEED+repeat-1)
        pred_store={(name,m):np.full(len(df),np.nan) for name,_,_,_ in outcome_specs() for m in ["Anatomy only","Anatomy + lesion-conditioned median burden"]}
        for fold,(tr,te) in enumerate(splitter.split(np.zeros(len(df)),groups),1):
            cont_tr=df.iloc[tr][PEDSQL_CONTINUOUS+DASS_CONTINUOUS].apply(pd.to_numeric,errors="coerce").to_numpy(float)
            cont_te=df.iloc[te][PEDSQL_CONTINUOUS+DASS_CONTINUOUS].apply(pd.to_numeric,errors="coerce").to_numpy(float)
            imp=SimpleImputer(strategy="median").fit(cont_tr); scale=StandardScaler().fit(imp.transform(cont_tr))
            ztr=scale.transform(imp.transform(cont_tr)); zte=scale.transform(imp.transform(cont_te))
            n_p=len(PEDSQL_CONTINUOUS); ctr=ztr[:,:n_p].mean(1)-ztr[:,n_p:].mean(1); cte=zte[:,:n_p].mean(1)-zte[:,n_p:].mean(1)
            btr=np.empty(len(tr),int); bte=np.empty(len(te),int)
            for fam in sorted(set(groups[tr])):
                a=np.where(groups[tr]==fam)[0]; b=np.where(groups[te]==fam)[0]; threshold=float(np.median(ctr[a]))
                btr[a]=(ctr[a]<threshold).astype(int); bte[b]=(cte[b]<threshold).astype(int)
            for name,col,binary,kind in outcome_specs():
                y=pd.to_numeric(df[col],errors="coerce").to_numpy(float); train=tr[np.isfinite(y[tr])]; test=te[np.isfinite(y[te])]
                if not len(test): continue
                trpos={idx:i for i,idx in enumerate(tr)}; tepos={idx:i for i,idx in enumerate(te)}
                for method,use in [("Anatomy only",False),("Anatomy + lesion-conditioned median burden",True)]:
                    train_b=np.array([btr[trpos[i]] for i in train]) if use else None; test_b=np.array([bte[tepos[i]] for i in test]) if use else None
                    m=model_pipeline(binary,burden=use); m.fit(frame(df.iloc[train],train_b),y[train]); p=m.predict_proba(frame(df.iloc[test],test_b))[:,1] if binary else m.predict(frame(df.iloc[test],test_b)); pred_store[(name,method)][test]=p
        for name,col,binary,kind in outcome_specs():
            y=pd.to_numeric(df[col],errors="coerce").to_numpy(float); ok=np.isfinite(y)
            for method in ["Anatomy only","Anatomy + lesion-conditioned median burden"]:
                p=pred_store[(name,method)][ok]; rows.append({"analysis":"lesion-conditioned median split","outcome":name,"validation_type":kind,"repeat":repeat,"method":method,"metric":"AUROC" if binary else "R2","performance":pooled_metric(y[ok],p,binary)})
    return pd.DataFrame(rows)


def partition_benchmark(df, assignments, repeats):
    # Full-cohort partitions are outcome-blind and held fixed. The participant fold grid is
    # the same lesion-family-stratified grid used by the burden benchmark.
    rows=[]
    strata=df["lesion_family_11"].astype(str).to_numpy()
    grids={}
    for repeat in range(1,repeats+1):
        sp=StratifiedKFold(N_SPLITS,shuffle=True,random_state=CV_SEED+repeat-1)
        grids[repeat]=list(sp.split(np.zeros(len(df)),strata))
    for (family,content),sub in assignments.groupby(["family","content_set"],sort=False):
        part=sub.set_index("record_id").loc[df.record_id,"cluster"].astype(str).to_numpy()
        label=f"{family} | {content}"
        for name,col,binary,kind in outcome_specs():
            y=pd.to_numeric(df[col],errors="coerce").to_numpy(float); evaluable=np.isfinite(y)
            for repeat in range(1,repeats+1):
                base=np.full(len(df),np.nan); aug=np.full(len(df),np.nan)
                for tr,te in grids[repeat]:
                    tr_eval=tr[evaluable[tr]]; te_eval=te[evaluable[te]]
                    if not len(te_eval): continue
                    for include,target in [(False,base),(True,aug)]:
                        m=model_pipeline(binary,partition=include)
                        m.fit(frame(df.iloc[tr_eval],partition=part[tr_eval] if include else None),y[tr_eval])
                        p=m.predict_proba(frame(df.iloc[te_eval],partition=part[te_eval] if include else None))[:,1] if binary else m.predict(frame(df.iloc[te_eval],partition=part[te_eval] if include else None))
                        target[te_eval]=p
                idx=np.flatnonzero(evaluable)
                pb=pooled_metric(y[idx],base[idx],binary); pa=pooled_metric(y[idx],aug[idx],binary)
                rows.append({"analysis":"primary HDBSCAN partition","representation":label,"outcome":name,"validation_type":kind,"repeat":repeat,"metric":"AUROC" if binary else "R2","anatomy_performance":pb,"partition_performance":pa,"delta_vs_anatomy":pa-pb})
    return pd.DataFrame(rows)

def lambda_burden(df, assignments, lambdas):
    pro=ordinal_pro_frame(df).to_numpy(float); ids=df.record_id.astype(str).to_numpy(); out=[]
    for lam in lambdas:
        sub=assignments[np.isclose(assignments["lambda"],lam)].set_index("record_id").loc[ids]
        clusters=sub.cluster.to_numpy(int); burden=np.full(len(df),np.nan)
        for cluster in sorted(c for c in np.unique(clusters) if c>=0):
            idx=np.where(clusters==cluster)[0]; raw=pro[idx]; imp=SimpleImputer(strategy="mean"); x=imp.fit_transform(raw)
            km=KMeans(2,n_init=KMEANS_N_INIT,random_state=KMEANS_SEED,algorithm="lloyd").fit(x); labels=km.labels_
            anchor=x[:,:len(PEDSQL_ORDINAL)].mean(1)-x[:,len(PEDSQL_ORDINAL):].mean(1); means={k:float(anchor[labels==k].mean()) for k in [0,1]}; higher=min(means,key=lambda k:(means[k],k)); burden[idx]=(labels==higher).astype(int)
        out.append(pd.DataFrame({"record_id":ids,"lambda":lam,"primary_cluster":clusters,"burden_binary":burden,"assigned":clusters>=0}))
    return pd.concat(out,ignore_index=True)


def lambda_clinical_robustness(df, burden, supported, repeats, full_grid=False):
    lo,hi=supported; lambdas=sorted(burden["lambda"].unique()); region=[x for x in lambdas if lo-1e-12<=x<=hi+1e-12]
    assigned_sets=[set(burden[(np.isclose(burden["lambda"],x))&burden.assigned].record_id.astype(str)) for x in region]; core=set.intersection(*assigned_sets)
    tracks=[("supported fixed-core",region,core)]
    if full_grid: tracks.append(("lambda-specific operating cohort",lambdas,None))
    rows=[]
    for track,ls,ids_core in tracks:
        for lam in ls:
            b=burden[np.isclose(burden["lambda"],lam)].set_index("record_id")
            ids=list(df.record_id.astype(str)) if ids_core is None else [x for x in df.record_id.astype(str) if x in ids_core]
            if ids_core is None: ids=[x for x in ids if bool(b.loc[x,"assigned"])]
            idx=np.array([df.index[df.record_id.astype(str)==x][0] for x in ids],int); feat=b.loc[ids,"burden_binary"].to_numpy(float)
            work=df.iloc[idx].reset_index(drop=True)
            for name,col,binary,kind in outcome_specs():
                yall=pd.to_numeric(work[col],errors="coerce"); eval_idx=np.flatnonzero(yall.notna()); y=yall.iloc[eval_idx].to_numpy(float); f=feat[eval_idx]
                for repeat in range(1,repeats+1):
                    seed=CV_SEED+repeat-1; sp=StratifiedKFold(N_SPLITS,shuffle=True,random_state=seed) if binary else KFold(N_SPLITS,shuffle=True,random_state=seed)
                    base=np.full(len(eval_idx),np.nan); aug=np.full(len(eval_idx),np.nan)
                    for trp,tep in (sp.split(eval_idx,y) if binary else sp.split(eval_idx)):
                        tr,te=eval_idx[trp],eval_idx[tep]
                        for use,target in [(False,base),(True,aug)]:
                            m=model_pipeline(binary,burden=use); m.fit(frame(work.iloc[tr],f[trp] if use else None),y[trp]); p=m.predict_proba(frame(work.iloc[te],f[tep] if use else None))[:,1] if binary else m.predict(frame(work.iloc[te],f[tep] if use else None)); target[tep]=p
                    pb=pooled_metric(y,base,binary); pa=pooled_metric(y,aug,binary)
                    rows.append({"track":track,"lambda":lam,"n_cohort":len(work),"outcome":name,"validation_type":kind,"repeat":repeat,"metric":"AUROC" if binary else "R2","anatomy_performance":pb,"burden_performance":pa,"delta_vs_anatomy":pa-pb})
    return pd.DataFrame(rows),pd.DataFrame({"record_id":sorted(core)})



def lambda_association_inference(df, burden, supported, core_ids):
    """Adjusted association and multiplicity analysis across the supported lambda interval."""
    lo,hi=supported
    region=[x for x in sorted(burden["lambda"].unique()) if lo-1e-12<=x<=hi+1e-12]
    core_set=set(core_ids)
    expected_direction={
        "6-minute walk distance":-1, "Cardiac anxiety":1, "Resilience":-1,
        "Trauma symptoms":1, "Sleep disturbance":1,
    }
    for name,_,binary,_ in outcome_specs():
        if binary: expected_direction[name]=1
    data_idx=df.set_index(df.record_id.astype(str),drop=False)
    rows=[]
    for lam in region:
        b=burden[np.isclose(burden["lambda"],lam) & burden.record_id.astype(str).isin(core_set)][["record_id","primary_cluster","burden_binary"]].copy()
        ids=[x for x in df.record_id.astype(str) if x in core_set]
        frame=data_idx.loc[ids].copy().reset_index(drop=True).merge(b,on="record_id",validate="one_to_one")
        for name,col,binary,kind in outcome_specs():
            work=frame[[col,"burden_binary","primary_cluster","age","sex","remoteness","chd_complexity"]].copy()
            work[col]=pd.to_numeric(work[col],errors="coerce"); work=work[work[col].notna() & work.burden_binary.notna()].reset_index(drop=True)
            work["primary_cluster"]=work.primary_cluster.astype("category")
            for c in ["sex","remoteness","chd_complexity"]: work[c]=work[c].astype("category")
            work["age"]=pd.to_numeric(work.age,errors="coerce"); work["age"]=work.age.fillna(work.age.median())
            rhs="burden_binary + age + C(sex) + C(remoteness) + C(chd_complexity) + C(primary_cluster)"
            if binary:
                work["_outcome"]=work[col].to_numpy(float)
                fit=smf.glm(f"_outcome ~ {rhs}",data=work,family=sm.families.Binomial()).fit(cov_type="HC3",maxiter=200)
                lower=work.copy(); higher=work.copy(); lower["burden_binary"]=0.; higher["burden_binary"]=1.
                effect=float(np.mean(fit.predict(higher)-fit.predict(lower))); ci=fit.conf_int().loc["burden_binary"]
                effect_name="adjusted_marginal_risk_difference_higher_minus_lower"
                events=int(np.sum(work._outcome.to_numpy(float)==1))
            else:
                y=work[col].to_numpy(float); sd=float(np.std(y,ddof=0))
                if not np.isfinite(sd) or sd<=0: raise RuntimeError(f"{name}, lambda={lam}: zero/non-finite outcome SD")
                work["_outcome"]=(y-y.mean())/sd
                fit=smf.ols(f"_outcome ~ {rhs}",data=work).fit(cov_type="HC3"); effect=float(fit.params["burden_binary"]); ci=fit.conf_int().loc["burden_binary"]
                effect_name="adjusted_standardised_difference_higher_minus_lower"; events=np.nan
            p=float(fit.pvalues["burden_binary"])
            rows.append({"lambda":lam,"outcome":name,"validation_type":kind,"outcome_type":"binary" if binary else "continuous","n":len(work),"events":events,
                         "effect_name":effect_name,"adjusted_effect":effect,"expected_adverse_direction":expected_direction[name],
                         "matches_expected_direction":bool(effect*expected_direction[name]>0),"burden_term_p_value":p,
                         "burden_term_model_scale_ci_low":float(ci.iloc[0]),"burden_term_model_scale_ci_high":float(ci.iloc[1])})
    inference=pd.DataFrame(rows)
    fdr=[]
    for lam,g in inference.groupby("lambda",sort=True):
        h=g.copy(); h["p_fdr_within_validation_family"]=np.nan
        for _,idx in h.groupby("validation_type").groups.items(): h.loc[list(idx),"p_fdr_within_validation_family"]=bh_adjust(h.loc[list(idx),"burden_term_p_value"])
        h["p_fdr_all_16"]=bh_adjust(h.burden_term_p_value); fdr.append(h)
    fdr=pd.concat(fdr,ignore_index=True)
    robust=[]
    for name,g in inference.groupby("outcome",sort=False):
        robust.append({"outcome":name,"validation_type":g.validation_type.iloc[0],"outcome_type":g.outcome_type.iloc[0],"n_lambdas":int(g["lambda"].nunique()),
                       "min_effect":float(g.adjusted_effect.min()),"max_effect":float(g.adjusted_effect.max()),"n_expected_direction":int(g.matches_expected_direction.sum()),
                       "all_expected_direction":bool(g.matches_expected_direction.all()),"min_raw_p_across_lambda":float(g.burden_term_p_value.min()),"max_raw_p_across_lambda":float(g.burden_term_p_value.max())})
    robust=pd.DataFrame(robust); robust["worst_case_p_fdr_within_validation_family"]=np.nan
    for _,idx in robust.groupby("validation_type").groups.items(): robust.loc[list(idx),"worst_case_p_fdr_within_validation_family"]=bh_adjust(robust.loc[list(idx),"max_raw_p_across_lambda"])
    robust["worst_case_p_fdr_all_16"]=bh_adjust(robust.max_raw_p_across_lambda)
    return inference,fdr,robust

def main():
    ap=argparse.ArgumentParser(description="Run transparent-burden, HDBSCAN-partition and lambda-robustness sensitivity analyses.")
    ap.add_argument("--data",required=True); ap.add_argument("--ordinal-labels",required=True)
    ap.add_argument("--cluster-dir",required=True,help="Output directory from script 04")
    ap.add_argument("--benchmark-dir",required=True,help="Output directory from script 05")
    ap.add_argument("--outdir",required=True); ap.add_argument("--repeats",type=int,default=N_REPEATS)
    ap.add_argument("--include-full-lambda-grid",action="store_true")
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df=load_data(args.data,args.ordinal_labels)
    median_split_benchmark(df,args.repeats).to_csv(out/"median_split_benchmark.csv",index=False)
    parts=pd.read_csv(Path(args.benchmark_dir)/"matched_hdbscan_assignments.csv.gz",dtype={"record_id":"string"}); partition_benchmark(df,parts,args.repeats).to_csv(out/"primary_partition_clinical_sensitivity.csv",index=False)
    manifest=read_json(Path(args.cluster_dir)/"selected_operating_point.json"); lo=float(manifest["hdbscan_sensitivity"]["supported_lambda_min"]); hi=float(manifest["hdbscan_sensitivity"]["supported_lambda_max"])
    ass=pd.read_csv(Path(args.cluster_dir)/"lambda_full_assignments.csv.gz",dtype={"record_id":"string"}); lambdas=sorted(ass["lambda"].unique()) if args.include_full_lambda_grid else [x for x in sorted(ass["lambda"].unique()) if lo-1e-12<=x<=hi+1e-12]
    lb=lambda_burden(df,ass,lambdas); lb.to_csv(out/"lambda_burden_assignments.csv.gz",index=False,compression="gzip")
    lr,core=lambda_clinical_robustness(df,lb,(lo,hi),args.repeats,args.include_full_lambda_grid); lr.to_csv(out/"lambda_clinical_robustness.csv",index=False); core.to_csv(out/"fixed_core_supported_lambda_participants.csv",index=False)
    inference,fdr,worst=lambda_association_inference(df,lb,(lo,hi),core["record_id"].astype(str).tolist())
    inference.to_csv(out/"lambda_adjusted_association_inference.csv",index=False)
    fdr.to_csv(out/"lambda_adjusted_association_fdr.csv",index=False)
    worst.to_csv(out/"lambda_adjusted_association_worst_case_fdr.csv",index=False)
    print(f"Sensitivity analyses complete; supported lambda interval {lo:.3f}-{hi:.3f}; outputs: {out}")

if __name__=="__main__": main()
