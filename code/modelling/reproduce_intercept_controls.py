"""Regenerate intercept-only leave-one-out controls from packaged outcomes."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
from metrics import rmse, r2
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/statistical-controls"

def load(path):
    with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))

def control(system, rows, id_col, outcome_col, outcome_name):
    obs=[]
    for row in rows:
        value=row.get(outcome_col,"")
        if value not in (None,""):
            try: obs.append((row.get(id_col,""),float(value)))
            except ValueError: pass
    if len(obs)<2: return None
    predictions=[]
    for i,(label,y) in enumerate(obs):
        pred=sum(v for j,(_,v) in enumerate(obs) if j!=i)/(len(obs)-1)
        predictions.append({"system":system,"outcome":outcome_name,"held_out":label,"observed":y,"loo_prediction":pred,"training_n":len(obs)-1})
    y=[v for _,v in obs]; p=[r["loo_prediction"] for r in predictions]
    return predictions,{"system":system,"outcome":outcome_name,"n":len(y),"rmse":rmse(y,p),"r2":r2(y,p),"prediction_rule":"mean of training outcomes in each leave-one-out fold"}

def main():
    outputs=[]; summaries=[]
    cu=load(ROOT/"data/cu-nhc/experimental-yields.csv")
    got=control("cu-nhc",cu,"catalyst","yield_percent","yield_percent")
    if got: outputs+=got[0]; summaries.append(got[1])
    au=load(ROOT/"data/au-cu/experimental-outcomes.csv")
    for col,name in [("yield_percent","yield_percent"),("de_percent","de_percent"),("ee_cis_percent","cis_ee_percent")]:
        got=control("au-cu",au,"catalyst",col,name)
        if got: outputs+=got[0]; summaries.append(got[1])
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"intercept-only-loo-predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["system","outcome","held_out","observed","loo_prediction","training_n"]);w.writeheader();w.writerows(outputs)
    (OUT/"intercept-only-summary.json").write_text(json.dumps({"method":"leave-one-out training-mean intercept baseline","metrics":"RMSE and R2 are computed on out-of-fold predictions; R2 denominator uses the full observed-response mean.","summaries":summaries},indent=2)+"\n")

if __name__=="__main__": main()
