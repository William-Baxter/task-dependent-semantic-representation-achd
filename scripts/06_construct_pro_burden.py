from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

from common import (
    CV_SEED, PEDSQL_CONTINUOUS, DASS_CONTINUOUS, PRO_CONTINUOUS, PRO_ORDINAL,
    add_lesion_family_11, bh_adjust, normalise_ids, ordinal_pro_frame,
)

METHODS = ["Semantic PRO", "Tabular PRO", "Continuous composite", "PC1"]
# Signature columns follow the empirically observed nested progression used in the article.
SIGNATURE_ORDER = ["Semantic PRO", "Tabular PRO", "PC1", "Continuous composite"]
N_SPLITS = 5
N_REPEATS = 10
KMEANS_N_INIT = 10
KMEANS_SEED = 42


def load_data(path: str, labels_path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"record_id":"string"}, low_memory=False)
    lab = pd.read_csv(labels_path, dtype={"record_id":"string"})
    df["record_id"] = normalise_ids(df["record_id"]); lab["record_id"] = normalise_ids(lab["record_id"])
    overlap = [c for c in PRO_ORDINAL if c in df]
    if overlap: df = df.drop(columns=overlap)
    df = df.merge(lab, on="record_id", how="inner", validate="one_to_one")
    df = add_lesion_family_11(df)
    required = {"record_id","chd_category","lesion_family_11","age","sex","remoteness","num_meds","num_comorb",
                "diabetes_label","hbp_label","asthma_label",*PRO_CONTINUOUS,*PRO_ORDINAL}
    missing = sorted(required-set(df.columns))
    if missing: raise ValueError("Missing fields: "+", ".join(missing))
    return df.sort_values("record_id", kind="mergesort").reset_index(drop=True)


def load_embedding(semantic_dir: Path, ids: pd.Series) -> np.ndarray:
    source_ids=[x.strip() for x in (semantic_dir.parent/"embedding_ids.txt").read_text().splitlines() if x.strip()]
    pos={x:i for i,x in enumerate(source_ids)}; order=[pos[str(x)] for x in ids]
    x=np.load(semantic_dir/"pro.npy")[order]
    if len(x)!=len(ids): raise RuntimeError("Semantic PRO alignment failed")
    return np.asarray(x,float)


class PROPreprocessor:
    def __init__(self):
        self.imp_ord=SimpleImputer(strategy="median"); self.scale_ord=StandardScaler()
        self.imp_cont=SimpleImputer(strategy="median"); self.scale_cont=StandardScaler()
        self.pca=PCA(n_components=1, svd_solver="full"); self.pc_sign=1.0
    def fit(self, df: pd.DataFrame):
        ord_raw=ordinal_pro_frame(df).to_numpy(float)
        self.ord=self.scale_ord.fit_transform(self.imp_ord.fit_transform(ord_raw))
        cont_raw=df[PRO_CONTINUOUS].apply(pd.to_numeric,errors="coerce").to_numpy(float)
        self.cont=self.scale_cont.fit_transform(self.imp_cont.fit_transform(cont_raw))
        n_p=len(PEDSQL_CONTINUOUS); self.anchor=self.cont[:,:n_p].mean(1)-self.cont[:,n_p:].mean(1)
        harm=self.cont.copy(); harm[:,n_p:]*=-1
        pc=self.pca.fit_transform(harm).ravel(); corr=np.corrcoef(pc,self.anchor)[0,1]
        self.pc_sign=-1.0 if np.isfinite(corr) and corr<0 else 1.0; self.pc=pc*self.pc_sign
        return self
    def transform(self, df: pd.DataFrame):
        o=self.scale_ord.transform(self.imp_ord.transform(ordinal_pro_frame(df).to_numpy(float)))
        c=self.scale_cont.transform(self.imp_cont.transform(df[PRO_CONTINUOUS].apply(pd.to_numeric,errors="coerce").to_numpy(float)))
        n_p=len(PEDSQL_CONTINUOUS); anchor=c[:,:n_p].mean(1)-c[:,n_p:].mean(1)
        harm=c.copy(); harm[:,n_p:]*=-1
        pc=self.pca.transform(harm).ravel()*self.pc_sign
        return o,c,anchor,pc


def matrices_full(df: pd.DataFrame, semantic_pro: np.ndarray):
    prep=PROPreprocessor().fit(df)
    return {
        "Semantic PRO": semantic_pro,
        "Tabular PRO": normalize(prep.ord, norm="l2"),
        "Continuous composite": prep.anchor[:,None],
        "PC1": prep.pc[:,None],
    }, prep.anchor


def fit_within_family(train_x, test_x, train_groups, test_groups, anchor):
    train_binary=np.empty(len(train_x),int); test_binary=np.empty(len(test_x),int)
    train_score=np.empty(len(train_x),float); test_score=np.empty(len(test_x),float)
    diagnostics=[]
    for family in sorted(set(train_groups)):
        tr=np.where(train_groups==family)[0]; te=np.where(test_groups==family)[0]
        if len(tr)<2 or len(np.unique(train_x[tr],axis=0))<2: raise RuntimeError(f"{family}: insufficient training variation for K=2")
        km=KMeans(n_clusters=2,n_init=KMEANS_N_INIT,random_state=KMEANS_SEED,algorithm="lloyd").fit(train_x[tr])
        raw_tr=km.labels_; raw_te=km.predict(test_x[te]) if len(te) else np.array([],int)
        means={k:float(np.mean(anchor[tr][raw_tr==k])) for k in [0,1]}
        higher=min(means,key=lambda k:(means[k],k)); lower=1-higher
        train_binary[tr]=(raw_tr==higher).astype(int); test_binary[te]=(raw_te==higher).astype(int)
        axis=km.cluster_centers_[higher]-km.cluster_centers_[lower]; norm=float(np.linalg.norm(axis))
        if norm<=0: raise RuntimeError(f"{family}: degenerate centroid axis")
        axis=axis/norm; midpoint=.5*(km.cluster_centers_[higher]+km.cluster_centers_[lower])
        raw_score=(train_x[tr]-midpoint)@axis; sd=float(np.std(raw_score,ddof=0))
        if sd<=0: raise RuntimeError(f"{family}: zero-variance burden axis")
        train_score[tr]=raw_score/sd; test_score[te]=((test_x[te]-midpoint)@axis)/sd if len(te) else np.array([])
        diagnostics.append({"lesion_family_11":family,"n_train":len(tr),"n_test":len(te),"anchor_mean_higher":means[higher],"anchor_mean_lower":means[lower]})
    return train_binary,test_binary,train_score,test_score,diagnostics


def full_data_burden(df, semantic_pro):
    mats, anchor=matrices_full(df,semantic_pro); groups=df["lesion_family_11"].astype(str).to_numpy()
    rows=[]; scores={}; assignments={}
    for method,x in mats.items():
        b,_,s,_,_=fit_within_family(x,np.empty((0,x.shape[1])),groups,np.array([],str),anchor)
        scores[method]=s; assignments[method]=b
        for rid,fam,bi,si in zip(df.record_id,groups,b,s): rows.append({"record_id":rid,"lesion_family_11":fam,"representation":method,"higher_burden":int(bi),"burden_score":float(si)})
    out=pd.DataFrame(rows)
    out["burden_score_z"]=out.groupby("representation")["burden_score"].transform(lambda x:(x-x.mean())/x.std(ddof=1))
    return out,scores,assignments


def oof_burden(df, semantic_pro, n_repeats):
    groups=df["lesion_family_11"].astype(str).to_numpy(); rows=[]; fold_rows=[]; diag=[]
    for repeat in range(1,n_repeats+1):
        splitter=StratifiedKFold(N_SPLITS,shuffle=True,random_state=CV_SEED+repeat-1)
        for fold,(tr,te) in enumerate(splitter.split(np.zeros(len(df)),groups),1):
            prep=PROPreprocessor().fit(df.iloc[tr]); otr,ctr,anchor,pc_tr=prep.ord,prep.cont,prep.anchor,prep.pc
            ote,cte,_,pc_te=prep.transform(df.iloc[te])
            matrices={
                "Semantic PRO":(semantic_pro[tr],semantic_pro[te]),
                "Tabular PRO":(normalize(otr),normalize(ote)),
                "Continuous composite":(anchor[:,None], (cte[:,:len(PEDSQL_CONTINUOUS)].mean(1)-cte[:,len(PEDSQL_CONTINUOUS):].mean(1))[:,None]),
                "PC1":(pc_tr[:,None],pc_te[:,None]),
            }
            for method,(xtr,xte) in matrices.items():
                btr,bte,strain,stest,d=fit_within_family(xtr,xte,groups[tr],groups[te],anchor)
                for x in d:
                    x.update({"repeat":repeat,"fold":fold,"representation":method})
                diag.extend(d)
                for pos,idx in enumerate(tr):
                    fold_rows.append({"record_id":df.at[idx,"record_id"],"repeat":repeat,"fold":fold,"split":"train","representation":method,"higher_burden":int(btr[pos]),"burden_score":float(strain[pos])})
                for pos,idx in enumerate(te):
                    rec={"record_id":df.at[idx,"record_id"],"repeat":repeat,"fold":fold,"representation":method,"higher_burden":int(bte[pos]),"burden_score":float(stest[pos])}
                    rows.append(rec)
                    fold_rows.append({**rec,"split":"test"})
    return pd.DataFrame(rows),pd.DataFrame(fold_rows),pd.DataFrame(diag)


def agreement_tables(full: pd.DataFrame):
    score=full.pivot(index="record_id",columns="representation",values="burden_score")
    binary=full.pivot(index="record_id",columns="representation",values="higher_burden")
    rows=[]
    for i,a in enumerate(METHODS):
        for b in METHODS[i+1:]: rows.append({"method_a":a,"method_b":b,"pearson_r":score[a].corr(score[b],method="pearson"),"binary_ari":adjusted_rand_score(binary[a],binary[b])})
    signature_matrix=binary[SIGNATURE_ORDER].astype(int)
    signature_codes=signature_matrix.astype(str).agg("".join,axis=1)
    signatures=signature_codes.value_counts().rename_axis("signature_raw").reset_index(name="n")
    signatures["proportion"]=signatures.n/len(binary)
    signatures["signature_order"]=" > ".join(SIGNATURE_ORDER)
    nested_labels={
        "0000":"None higher burden",
        "1000":"Semantic PRO only",
        "1100":"Semantic PRO + Tabular PRO",
        "1110":"Semantic PRO + Tabular PRO + PC1",
        "1111":"All four formulations",
    }
    signatures["dominant_nested_pattern"]=signatures["signature_raw"].map(nested_labels).fillna("Non-nested combination")
    # Prefix the binary code so generic CSV readers cannot coerce leading-zero signatures to integers.
    signatures.insert(0,"signature", "S"+signatures["signature_raw"])
    for method in SIGNATURE_ORDER:
        signatures[method.replace(" ", "_").replace("/", "_").lower()]=signatures["signature_raw"].str[SIGNATURE_ORDER.index(method)].astype(int)
    signatures=signatures.drop(columns="signature_raw")
    return pd.DataFrame(rows),signatures,binary


def encode_binary_predictor(series: pd.Series, positive: str, negative: str):
    s=series.astype("string").str.strip(); out=pd.Series(np.nan,index=series.index,dtype=float)
    out[s.eq(positive)]=1.; out[s.eq(negative)]=0.; return out


def external_characteristics(df: pd.DataFrame, full: pd.DataFrame):
    wide_b=full.pivot(index="record_id",columns="representation",values="higher_burden").loc[df.record_id]
    wide_s=full.pivot(index="record_id",columns="representation",values="burden_score_z").loc[df.record_id]
    predictors={
        "Age (per 10 years)":pd.to_numeric(df.age,errors="coerce")/10,
        "Female sex":encode_binary_predictor(df.sex,"Female","Male"),
        "Regional/remote residence":encode_binary_predictor(df.remoteness,"Regional/Remote","Urban"),
        "Medication count (per medication)":pd.to_numeric(df.num_meds,errors="coerce"),
        "Comorbidity count (per comorbidity)":pd.to_numeric(df.num_comorb,errors="coerce"),
        "Diabetes":encode_binary_predictor(df.diabetes_label,"Yes","No"),
        "Hypertension":encode_binary_predictor(df.hbp_label,"Yes","No"),
        "Asthma":encode_binary_predictor(df.asthma_label,"Yes","No"),
    }
    binary_rows=[]; continuous_rows=[]
    lesion=df.chd_category.astype("string").str.strip().to_numpy()
    for method in METHODS:
        for name,x in predictors.items():
            m=pd.DataFrame({"y":wide_b[method].to_numpy(float),"x":x.to_numpy(float),"lesion":lesion}).dropna()
            fit=smf.glm("y ~ x + C(lesion)",data=m,family=sm.families.Binomial()).fit()
            beta=float(fit.params["x"]); se=float(fit.bse["x"]); z=1.959963984540054
            binary_rows.append({"representation":method,"characteristic":name,"n_model":len(m),"odds_ratio":np.exp(beta),"ci95_low":np.exp(beta-z*se),"ci95_high":np.exp(beta+z*se),"p_value":float(fit.pvalues["x"])})
        # Continuous burden external-gradient analysis uses the five prespecified characteristics.
        for name in list(predictors)[:5]:
            x=predictors[name]; m=pd.DataFrame({"y":wide_s[method].to_numpy(float),"x":x.to_numpy(float),"lesion":lesion}).dropna()
            fit=smf.ols("y ~ x + C(lesion)",data=m).fit(); ci=fit.conf_int().loc["x"]
            continuous_rows.append({"representation":method,"characteristic":name,"n_model":len(m),"beta":float(fit.params["x"]),"ci95_low":float(ci.iloc[0]),"ci95_high":float(ci.iloc[1]),"p_value":float(fit.pvalues["x"])})
    br=pd.DataFrame(binary_rows); cr=pd.DataFrame(continuous_rows)
    br["p_fdr_bh"]=br.groupby("representation")["p_value"].transform(lambda x:bh_adjust(x))
    cr["p_fdr_bh"]=cr.groupby("representation")["p_value"].transform(lambda x:bh_adjust(x))
    return br,cr


def main():
    ap=argparse.ArgumentParser(description="Construct four full-cohort within-lesion PRO burden formulations and fold-local OOF reconstructions.")
    ap.add_argument("--data",required=True); ap.add_argument("--ordinal-labels",required=True)
    ap.add_argument("--semantic-dir",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--repeats",type=int,default=N_REPEATS)
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    df=load_data(args.data,args.ordinal_labels); semantic=load_embedding(Path(args.semantic_dir),df.record_id)
    full,_,_=full_data_burden(df,semantic); full.to_csv(out/"full_cohort_burden_assignments.csv",index=False)
    oof,fold_features,diag=oof_burden(df,semantic,args.repeats); oof.to_csv(out/"oof_burden_features.csv.gz",index=False,compression="gzip"); fold_features.to_csv(out/"fold_burden_features.csv.gz",index=False,compression="gzip"); diag.to_csv(out/"oof_burden_diagnostics.csv.gz",index=False,compression="gzip")
    agreement,signatures,binary=agreement_tables(full); agreement.to_csv(out/"cross_representation_agreement.csv",index=False); signatures.to_csv(out/"higher_burden_signatures.csv",index=False)
    # OOF binary reconstruction stability against the full-data assignment, one ARI per repeat/method.
    full_binary=full.pivot(index="record_id",columns="representation",values="higher_burden")
    oof_binary=oof.pivot_table(index="record_id",columns=["representation","repeat"],values="higher_burden")
    stability=[]
    for method in METHODS:
        for repeat in range(1,args.repeats+1):
            aligned=oof_binary[(method,repeat)].loc[full_binary.index]
            stability.append({"representation":method,"repeat":repeat,"assignment_ari":adjusted_rand_score(full_binary[method],aligned),"raw_agreement":float(np.mean(full_binary[method].to_numpy()==aligned.to_numpy()))})
    pd.DataFrame(stability).to_csv(out/"oof_burden_reconstruction_stability.csv",index=False)
    br,cr=external_characteristics(df,full); br.to_csv(out/"external_characteristics_binary.csv",index=False); cr.to_csv(out/"external_characteristics_continuous.csv",index=False)
    print(f"Burden construction complete for {len(df)} participants; outputs: {out}")

if __name__=="__main__": main()
