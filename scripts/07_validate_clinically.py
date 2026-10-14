from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from scipy.linalg import LinAlgWarning
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    BASE_NUMERIC, BASE_CATEGORICAL, CONTEXT_FIELDS, PRO_ORDINAL, PRO_CONTINUOUS,
    PEDSQL_ORDINAL, DASS_ORDINAL, PEDSQL_ORDER, DASS_ORDER,
    derive_binary_outcomes, normalise_ids, one_hot_encoder, ordinal_pro_frame,
    outcome_specs, CONVERGENT_OUTCOMES, bh_adjust,
)

N_SPLITS=5
N_REPEATS=10
OUTER_SEED=20260904
INNER_SEED=20260904
ALPHAS=np.logspace(-4,4,9)
CS=np.logspace(-4,4,9)
ASSOCIATION_DRAWS=1500
BURDEN_METHODS=["Semantic PRO","Tabular PRO","Continuous composite","PC1"]


def load_data(path, labels_path):
    df=pd.read_csv(path,dtype={"record_id":"string"},low_memory=False)
    labels=pd.read_csv(labels_path,dtype={"record_id":"string"})
    df["record_id"]=normalise_ids(df.record_id); labels["record_id"]=normalise_ids(labels.record_id)
    overlap=[c for c in PRO_ORDINAL if c in df]
    if overlap: df=df.drop(columns=overlap)
    df=df.merge(labels,on="record_id",how="inner",validate="one_to_one")
    df=derive_binary_outcomes(df)
    if "age_band" not in df:
        age=pd.to_numeric(df["age"],errors="coerce")
        df["age_band"]=pd.cut(age,[-np.inf,29,39,49,59,np.inf],labels=["18-29","30-39","40-49","50-59","60+"]).astype("string")
    required={"record_id","age","sex","remoteness","chd_complexity","chd_category",*PRO_CONTINUOUS,*PRO_ORDINAL}
    for _,col,_,_ in outcome_specs(): required.add(col)
    missing=sorted(required-set(df.columns))
    if missing: raise ValueError("Missing fields: "+", ".join(missing))
    return df.sort_values("record_id",kind="mergesort").reset_index(drop=True)


def clean_cats(frame, columns):
    out=frame[columns].copy()
    for c in columns:
        s=out[c].astype("string").str.strip(); out[c]=s.astype(object).where(s.notna(),np.nan)
    return out


def anatomy_frame(df, burden=None, include_raw_pro=False):
    cols=BASE_NUMERIC+BASE_CATEGORICAL+(PRO_CONTINUOUS if include_raw_pro else [])
    x=df[cols].copy()
    if burden is not None: x["burden_feature"]=np.asarray(burden,float)
    return x


def outcome_pipeline(binary, include_burden=False, include_raw_pro=False):
    numeric=list(BASE_NUMERIC)+(PRO_CONTINUOUS if include_raw_pro else [])+(["burden_feature"] if include_burden else [])
    prep=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",one_hot_encoder())]),BASE_CATEGORICAL),
    ],remainder="drop")
    model=LogisticRegression(C=1e6,solver="liblinear",max_iter=5000,random_state=OUTER_SEED) if binary else LinearRegression()
    return Pipeline([("prep",prep),("model",model)])


def spearman_audit(df):
    validators=[(name,col,binary,kind) for name,col,binary,kind in outcome_specs()]
    rows=[]
    for pro in PRO_CONTINUOUS:
        x=pd.to_numeric(df[pro],errors="coerce")
        for name,col,binary,kind in validators:
            y=pd.to_numeric(df[col],errors="coerce"); ok=x.notna()&y.notna()
            rho=x[ok].corr(y[ok],method="spearman") if ok.sum()>=3 else np.nan
            rows.append({"construction_pro":pro,"validator":name,"validation_type":kind,"validator_type":"binary" if binary else "continuous","n":int(ok.sum()),"spearman_rho":rho})
    return pd.DataFrame(rows)


def adjusted_associations(df, full_assignments, draws=ASSOCIATION_DRAWS):
    wide=full_assignments.pivot(index="record_id",columns="representation",values="higher_burden").loc[df.record_id]
    rows=[]; rng=np.random.default_rng(20260906)
    for method in BURDEN_METHODS:
        burden=wide[method].to_numpy(float)
        for name,col,binary,kind in outcome_specs():
            y=pd.to_numeric(df[col],errors="coerce")
            model=pd.DataFrame({"y":y,"burden":burden,"age":pd.to_numeric(df.age,errors="coerce"),
                                "sex":df.sex,"remoteness":df.remoteness,"complexity":df.chd_complexity,"diagnosis":df.chd_category}).dropna()
            if binary:
                fit=smf.glm("y ~ burden + age + C(sex) + C(remoteness) + C(complexity) + C(diagnosis)",data=model,family=sm.families.Binomial()).fit()
                lo=model.copy(); hi=model.copy(); lo["burden"]=0.; hi["burden"]=1.
                rd=float(np.mean(fit.predict(hi)-fit.predict(lo)))
                beta=np.asarray(fit.params,float); cov=np.asarray(fit.cov_params(),float)
                sampled=rng.multivariate_normal(beta,cov,size=draws,check_valid="ignore")
                import patsy
                rhs = "burden + age + C(sex) + C(remoteness) + C(complexity) + C(diagnosis)"
                exog_lo = patsy.dmatrix(rhs, lo, return_type="dataframe")
                exog_hi = patsy.dmatrix(rhs, hi, return_type="dataframe")
                # Align explicitly to the fitted coefficient order. This avoids relying on
                # statsmodels' private formula/design-info storage, which differs across versions.
                exog_lo = np.asarray(exog_lo.reindex(columns=fit.params.index, fill_value=0.0), float)
                exog_hi = np.asarray(exog_hi.reindex(columns=fit.params.index, fill_value=0.0), float)
                def expit(z): return 1/(1+np.exp(-np.clip(z,-40,40)))
                draw_rd=(expit(exog_hi@sampled.T)-expit(exog_lo@sampled.T)).mean(axis=0)
                ci=np.quantile(draw_rd,[.025,.975])
                rows.append({"representation":method,"outcome":name,"validation_type":kind,"outcome_type":"binary","n":len(model),"effect":"adjusted risk difference","estimate":rd,"ci95_low":float(ci[0]),"ci95_high":float(ci[1]),"p_value":float(fit.pvalues["burden"])})
            else:
                model["y_z"]=(model.y-model.y.mean())/model.y.std(ddof=0)
                fit=smf.ols("y_z ~ burden + age + C(sex) + C(remoteness) + C(complexity) + C(diagnosis)",data=model).fit(); ci=fit.conf_int().loc["burden"]
                rows.append({"representation":method,"outcome":name,"validation_type":kind,"outcome_type":"continuous","n":len(model),"effect":"adjusted difference (SD)","estimate":float(fit.params["burden"]),"ci95_low":float(ci.iloc[0]),"ci95_high":float(ci.iloc[1]),"p_value":float(fit.pvalues["burden"])})
    out=pd.DataFrame(rows)
    out["p_fdr_bh"]=np.nan
    for _,idx in out.groupby("representation",sort=False).groups.items():
        idx=list(idx)
        out.loc[idx,"p_fdr_bh"]=bh_adjust(out.loc[idx,"p_value"].to_numpy(float))
    out["fdr_sig"]=out["p_fdr_bh"]<0.05
    return out


def burden_oof_performance(df, fold_features):
    rows=[]; prediction_rows=[]
    for repeat in sorted(fold_features.repeat.unique()):
        rep=fold_features[fold_features.repeat==repeat]
        for fold in sorted(rep.fold.unique()):
            fold_df=rep[rep.fold==fold]
            for method in BURDEN_METHODS:
                mf=fold_df[fold_df.representation==method]
                train_map=mf[mf.split=="train"].set_index("record_id")
                test_map=mf[mf.split=="test"].set_index("record_id")
                tr_idx=np.flatnonzero(df.record_id.isin(train_map.index)); te_idx=np.flatnonzero(df.record_id.isin(test_map.index))
                train_ids=df.iloc[tr_idx].record_id; test_ids=df.iloc[te_idx].record_id
                for name,col,binary,kind in outcome_specs():
                    y=pd.to_numeric(df[col],errors="coerce").to_numpy(float); tr=tr_idx[np.isfinite(y[tr_idx])]; te=te_idx[np.isfinite(y[te_idx])]
                    if not len(te): continue
                    # Both train/test burden features came from the same training-only construction for this fold.
                    btr=train_map.loc[df.iloc[tr].record_id,"higher_burden"].to_numpy(float); bte=test_map.loc[df.iloc[te].record_id,"higher_burden"].to_numpy(float)
                    strn=train_map.loc[df.iloc[tr].record_id,"burden_score"].to_numpy(float); ste=test_map.loc[df.iloc[te].record_id,"burden_score"].to_numpy(float)
                    for feature_name,ftr,fte in [("binary",btr,bte),("axis",strn,ste)]:
                        model=outcome_pipeline(binary,include_burden=True)
                        xtr=anatomy_frame(df.iloc[tr],ftr); xte=anatomy_frame(df.iloc[te],fte); model.fit(xtr,y[tr]); pred=model.predict_proba(xte)[:,1] if binary else model.predict(xte)
                        for idx,p in zip(te,pred): prediction_rows.append({"record_id":df.at[idx,"record_id"],"outcome":name,"validation_type":kind,"repeat":repeat,"fold":fold,"method":f"Anatomy + {method} [{feature_name}]","truth":y[idx],"prediction":float(p)})
            # anatomy baseline is identical across burden methods; run once per outcome/fold.
            train_ids=fold_df[fold_df.split=="train"].record_id.unique(); test_ids=fold_df[fold_df.split=="test"].record_id.unique()
            tr_idx=np.flatnonzero(df.record_id.isin(train_ids)); te_idx=np.flatnonzero(df.record_id.isin(test_ids))
            for name,col,binary,kind in outcome_specs():
                y=pd.to_numeric(df[col],errors="coerce").to_numpy(float); tr=tr_idx[np.isfinite(y[tr_idx])]; te=te_idx[np.isfinite(y[te_idx])]
                if not len(te): continue
                model=outcome_pipeline(binary); model.fit(anatomy_frame(df.iloc[tr]),y[tr]); pred=model.predict_proba(anatomy_frame(df.iloc[te]))[:,1] if binary else model.predict(anatomy_frame(df.iloc[te]))
                for idx,p in zip(te,pred): prediction_rows.append({"record_id":df.at[idx,"record_id"],"outcome":name,"validation_type":kind,"repeat":repeat,"fold":fold,"method":"Anatomy only","truth":y[idx],"prediction":float(p)})
    preds=pd.DataFrame(prediction_rows)
    for keys,g in preds.groupby(["outcome","validation_type","repeat","method"],sort=False):
        outcome,kind,repeat,method=keys; binary=outcome in {x[0] for x in outcome_specs() if x[2]}
        y=g.truth.to_numpy(float); p=g.prediction.to_numpy(float); perf=roc_auc_score(y,p) if binary else r2_score(y,p)
        rows.append({"outcome":outcome,"validation_type":kind,"repeat":repeat,"method":method,"metric":"AUROC" if binary else "R2","n_evaluable":len(g),"performance":float(perf)})
    return pd.DataFrame(rows),preds


class VignetteTabular:
    def __init__(self):
        self.ctx_imp=SimpleImputer(strategy="most_frequent"); self.ctx_ohe=one_hot_encoder(dense=True)
        self.pro_imp=SimpleImputer(strategy="mean"); self.pro_scaler=StandardScaler()
    def fit(self,df):
        c=self.ctx_imp.fit_transform(clean_cats(df,CONTEXT_FIELDS)); self.ctx_ohe.fit(c)
        p=self.pro_imp.fit_transform(ordinal_pro_frame(df)); self.pro_scaler.fit(p); return self
    def transform(self,df):
        c=self.ctx_ohe.transform(self.ctx_imp.transform(clean_cats(df,CONTEXT_FIELDS)))
        p=self.pro_scaler.transform(self.pro_imp.transform(ordinal_pro_frame(df)))
        return np.c_[np.asarray(c,float),np.asarray(p,float)]


def load_static_semantics(semantic_root: Path, ids: pd.Series):
    src=[x.strip() for x in (semantic_root/"embedding_ids.txt").read_text().splitlines() if x.strip()]; pos={x:i for i,x in enumerate(src)}; order=[pos[str(x)] for x in ids]
    reps={}
    for enc,label in [("adapted","Semantic adapted"),("base","Semantic base")]:
        d=semantic_root/enc
        if d.exists():
            reps[f"{label} joint"]=np.load(d/"joint.npy")[order]
            reps[f"{label} separate"]=np.load(d/"separate_equal.npy")[order]
    return reps


def direct_probe_benchmark(df, semantic_root: Path, repeats):
    static=load_static_semantics(semantic_root,df.record_id)
    principal=["Semantic adapted joint","Semantic adapted separate","Tabular additive","Anatomy + raw continuous PROs"]
    for m in principal[:2]:
        if m not in static: raise RuntimeError(f"Required representation missing: {m}")
    methods=principal+[m for m in ["Semantic base joint","Semantic base separate"] if m in static]
    rows=[]; preds=[]
    for name,col,binary,kind in outcome_specs():
        yall=pd.to_numeric(df[col],errors="coerce"); eval_idx=np.flatnonzero(yall.notna()); y=yall.iloc[eval_idx].to_numpy(float)
        for repeat in range(1,repeats+1):
            seed=OUTER_SEED+repeat-1
            splitter=StratifiedKFold(N_SPLITS,shuffle=True,random_state=seed) if binary else KFold(N_SPLITS,shuffle=True,random_state=seed)
            splits=list(splitter.split(eval_idx,y) if binary else splitter.split(eval_idx))
            oof={m:np.full(len(eval_idx),np.nan) for m in methods}
            for fold,(trpos,tepos) in enumerate(splits,1):
                tr=eval_idx[trpos]; te=eval_idx[tepos]
                for method in methods:
                    if method in static:
                        xtr,xte=static[method][tr],static[method][te]
                    elif method=="Tabular additive":
                        b=VignetteTabular().fit(df.iloc[tr]); xtr,xte=b.transform(df.iloc[tr]),b.transform(df.iloc[te])
                    else:
                        b=AnatomyRawBuilder().fit(df.iloc[tr]); xtr,xte=b.transform(df.iloc[tr]),b.transform(df.iloc[te])
                    scaler=StandardScaler(); xtr=scaler.fit_transform(xtr); xte=scaler.transform(xte)
                    inner_seed=INNER_SEED+10000*repeat+fold
                    if binary:
                        inner=StratifiedKFold(5,shuffle=True,random_state=inner_seed)
                        model=LogisticRegressionCV(
                            Cs=CS,cv=inner,scoring="roc_auc",solver="liblinear",
                            l1_ratios=(0.0,),use_legacy_attributes=False,
                            max_iter=5000,n_jobs=-1,
                        ).fit(xtr,y[trpos])
                        pred=model.predict_proba(xte)[:,1]
                    else:
                        inner=KFold(5,shuffle=True,random_state=inner_seed)
                        # Very weak candidate penalties can be numerically ill-conditioned
                        # in high-dimensional semantic spaces. RidgeCV evaluates them as
                        # part of the prespecified grid; finite-output checks below guard
                        # the selected fit while suppressing only the known linear-algebra
                        # warning generated by rejected/near-singular candidates.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", LinAlgWarning)
                            model=RidgeCV(alphas=ALPHAS,cv=inner,scoring="neg_mean_squared_error").fit(xtr,y[trpos])
                        if not np.isfinite(float(model.alpha_)):
                            raise RuntimeError(f"Non-finite selected Ridge alpha for {name} / {method}")
                        pred=model.predict(xte)
                        if not np.isfinite(pred).all():
                            raise RuntimeError(f"Non-finite Ridge predictions for {name} / {method}")
                    oof[method][tepos]=pred
                    for pos,p in zip(tepos,pred): preds.append({"record_id":df.at[eval_idx[pos],"record_id"],"outcome":name,"validation_type":kind,"repeat":repeat,"fold":fold,"method":method,"truth":y[pos],"prediction":float(p)})
            for method,p in oof.items():
                perf=roc_auc_score(y,p) if binary else r2_score(y,p)
                rows.append({"outcome":name,"validation_type":kind,"repeat":repeat,"method":method,"metric":"AUROC" if binary else "R2","n_evaluable":len(y),"performance":float(perf)})
    return pd.DataFrame(rows),pd.DataFrame(preds)


class AnatomyRawBuilder:
    def __init__(self):
        self.num=BASE_NUMERIC+PRO_CONTINUOUS; self.nimp=SimpleImputer(strategy="median"); self.cimp=SimpleImputer(strategy="most_frequent"); self.ohe=one_hot_encoder(dense=True)
    def fit(self,df):
        self.nimp.fit(df[self.num].apply(pd.to_numeric,errors="coerce")); c=self.cimp.fit_transform(clean_cats(df,BASE_CATEGORICAL)); self.ohe.fit(c); return self
    def transform(self,df):
        n=self.nimp.transform(df[self.num].apply(pd.to_numeric,errors="coerce")); c=self.ohe.transform(self.cimp.transform(clean_cats(df,BASE_CATEGORICAL))); return np.c_[n,np.asarray(c,float)]


def main():
    ap=argparse.ArgumentParser(description="Evaluate burden formulations and full-summary representations against held-out clinical and functional outcomes.")
    ap.add_argument("--data",required=True); ap.add_argument("--ordinal-labels",required=True)
    ap.add_argument("--burden-dir",required=True); ap.add_argument("--semantic-root",required=True)
    ap.add_argument("--outdir",required=True); ap.add_argument("--repeats",type=int,default=N_REPEATS)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df=load_data(args.data,args.ordinal_labels)
    burden_dir=Path(args.burden_dir); full=pd.read_csv(burden_dir/"full_cohort_burden_assignments.csv",dtype={"record_id":"string"}); fold=pd.read_csv(burden_dir/"fold_burden_features.csv.gz",dtype={"record_id":"string"})
    spearman_audit(df).to_csv(out/"construction_pro_validator_spearman.csv",index=False)
    adjusted_associations(df,full).to_csv(out/"adjusted_burden_outcome_associations.csv",index=False)
    bm,bpred=burden_oof_performance(df,fold); bm.to_csv(out/"burden_oof_performance.csv",index=False); bpred.to_csv(out/"burden_oof_predictions.csv.gz",index=False,compression="gzip")
    dm,dpred=direct_probe_benchmark(df,Path(args.semantic_root),args.repeats); dm.to_csv(out/"direct_probe_oof_performance.csv",index=False); dpred.to_csv(out/"direct_probe_oof_predictions.csv.gz",index=False,compression="gzip")
    print(f"Clinical validation complete; outputs: {out}")

if __name__=="__main__": main()
