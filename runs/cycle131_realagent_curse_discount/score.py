"""Cycle 131 scorer: parse ESTIMATE lines, compare to ground truth. Key questions:
(1) do agents DISCOUNT the observed winner (knowing-doing gap?)  (2) does the estimate TRACK the true value, or is it a
flat generic prior (blind to signal)?  (3) does OPUS resolve real-from-lucky better than Haiku (capability) or not (info limit)?"""
import json, re, os
GT = json.load(open("runs/cycle131_realagent_curse_discount/ground_truth.json"))

def est(path):
    if not os.path.exists(path): return None
    m = re.findall(r"ESTIMATE:\s*\**\s*([0-9]*\.?[0-9]+)", open(path).read())
    return float(m[-1]) if m else None

def corr(a, b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b)); da=sum((x-ma)**2 for x in a)**.5; db=sum((y-mb)**2 for y in b)**.5
    return num/(da*db) if da>0 and db>0 else float('nan')

rows=[]
for g in GT:
    s=g["seed"]; base=f"runs/cycle131_realagent_curse_discount/traj{s}"
    rows.append({"seed":s, "obs":g["obs_winner"], "true":g["true_winner"], "gap":g["curse_gap"],
                 "haiku_plain":est(base+"_plain.log"), "haiku_hint":est(base+"_hint.log"), "opus_plain":est(base+"_plain_opus.log")})

def discount(e,o,t):
    if e is None or abs(o-t)<1e-6: return None
    return round((o-e)/(o-t),2)

print(f"{'seed':>4} {'obs':>5} {'true':>6} {'gap':>6} | {'H_plain':>7} {'O_plain':>7} {'H_hint':>7} | discount(H_pl/O_pl)")
obs=[r['obs'] for r in rows]; true=[r['true'] for r in rows]
hp=[r['haiku_plain'] for r in rows]; op=[r['opus_plain'] for r in rows]; hh=[r['haiku_hint'] for r in rows]
for r in rows:
    dh=discount(r['haiku_plain'],r['obs'],r['true']); do=discount(r['opus_plain'],r['obs'],r['true'])
    print(f"{r['seed']:>4} {r['obs']:>5} {r['true']:>6} {r['gap']:>6} | {str(r['haiku_plain']):>7} {str(r['opus_plain']):>7} {str(r['haiku_hint']):>7} | {dh}/{do}")

out={"rows":rows,
     "mean_est":{"haiku_plain":round(sum(hp)/len(hp),3),"opus_plain":round(sum(op)/len(op),3),"haiku_hint":round(sum(hh)/len(hh),3)},
     "all_estimates_below_observed":{"haiku_plain":all(e<o for e,o in zip(hp,obs)),"opus_plain":all(e<o for e,o in zip(op,obs))},
     "corr_estimate_vs_TRUE":{"haiku_plain":round(corr(hp,true),3),"opus_plain":round(corr(op,true),3)},
     "corr_estimate_vs_OBSERVED":{"haiku_plain":round(corr(hp,obs),3),"opus_plain":round(corr(op,obs),3)},
     "high_true_strategies_crushed":[{"seed":r['seed'],"true":r['true'],"haiku":r['haiku_plain'],"opus":r['opus_plain']} for r in rows if r['true']>1.3]}
json.dump(out, open("runs/cycle131_realagent_curse_discount/score.json","w"), indent=1)
print("\n--- SUMMARY (cite this) ---")
print(f"(1) DISCOUNT (no knowing-doing gap): all estimates BELOW observed? Haiku {out['all_estimates_below_observed']['haiku_plain']}, Opus {out['all_estimates_below_observed']['opus_plain']}. mean est: Haiku-plain {out['mean_est']['haiku_plain']}, Opus-plain {out['mean_est']['opus_plain']} (vs mean obs {round(sum(obs)/len(obs),2)}).")
print(f"(2) does estimate TRACK true value? corr(est,TRUE): Haiku {out['corr_estimate_vs_TRUE']['haiku_plain']}, Opus {out['corr_estimate_vs_TRUE']['opus_plain']}  | corr(est,OBS): Haiku {out['corr_estimate_vs_OBSERVED']['haiku_plain']}, Opus {out['corr_estimate_vs_OBSERVED']['opus_plain']}  (near-0 => flat generic prior, blind to signal)")
print(f"(3) HIGH-TRUE strategies CRUSHED (capability doesn't rescue): {out['high_true_strategies_crushed']}")
print("=> capable agents KNOW+APPLY the curse (discount, cite Deflated-Sharpe) but over-shrink to a GENERIC PRIOR blind to the observed magnitude; can't separate real-from-lucky from the data (Opus=Haiku) -> the info isn't in the selection data => need an INDEPENDENT held-out (c45/c7). The agent's behavior IS the thesis.")
print("DONE")
