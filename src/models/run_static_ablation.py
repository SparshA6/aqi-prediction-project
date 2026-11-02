"""Compare environmental inputs with physically meaningful station covariates."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from baselines import fit_predict
from data import SAT, WX, TEMPORAL, load_dataset, make_splits
from evaluate import evaluate
from paths import ROOT
from runs import add_run_args, start_run, complete_run, select_splits


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol",choices=["chrono","loco","season_cv"],default="loco")
    ap.add_argument("--models",nargs="+",choices=["Ridge","XGBoost"],default=["Ridge","XGBoost"])
    ap.add_argument("--quick",action="store_true")
    add_run_args(ap)
    args=ap.parse_args()
    df=load_dataset()
    cov=pd.read_csv(ROOT/"data/processed/station_covariates/covariates.csv")
    cols=[c for c in cov if c!="station_id"]
    df=df.merge(cov,on="station_id",how="left",validate="many_to_one")
    if df[cols].isna().any().any():raise ValueError("Missing station covariates")
    if args.quick:df=df.sample(4000,random_state=0).sort_values(["date","station_id"]).reset_index(drop=True)
    sets={"ENV-T":SAT+WX+TEMPORAL,"ENV-T+P":SAT+WX+TEMPORAL+cols}
    args.sets=list(sets)
    splits=select_splits(make_splits(df,args.protocol),args.folds)
    out=start_run(args,"static",df,splits)
    metrics=[]
    for fold,tr,evs in splits:
        for name,features in sets.items():
            for model in args.models:
                prediction=fit_predict(model,df.loc[tr,features],df.loc[tr,"y"],{k:df.loc[i,features] for k,i in evs.items()},n_jobs=2)
                for split,index in evs.items():
                    for row in evaluate(df.loc[index],prediction[split]):
                        metrics.append(dict(protocol=args.protocol,fold=fold,split=split,feature_set=name,model=model,**row))
                print(f"{fold} {name} {model} complete",flush=True)
        pd.DataFrame(metrics).to_csv(out/f"metrics_{args.protocol}.csv",index=False)
    complete_run(out)
    print(f"Static feature ablation saved: {out}")


if __name__=="__main__":main()
