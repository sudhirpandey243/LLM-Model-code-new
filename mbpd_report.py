#!/usr/bin/env python
"""Summarise the v18.10/v18.11 + DICD MBPD diagnostic stored in results.json.

usage:  python mbpd_report.py /path/to/results.json

Supports:
  1) Per-example held-out (eval) frame readout with family and cf_op ('mirror'
     or 'dual').
  2) Pooled hypothesis test across all examples and per relation family.
  3) Continuous correlation analysis: pre-attack perception P (both active
     P_signed and clean-only Proposition-Dual P_dual) vs. final adversarial
     margin on true-question ('No') rows.
  4) 3-Way Orthogonal Decomposition on lateral examples (Frame Assent Bias A_f,
     Lexical Left/Right Word Prior L_f, and Pure Visual Perception P_f), plus
     empirical validation of Proposition-Dual Inversion (corr(P_dual, P_mirror)).
"""
import json, sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
res = json.load(open(path))
res = [r for r in res if r.get("mbpd_diag")]
if not res:
    sys.exit("no mbpd_diag in %s (was MBPD_DIAG on?)" % path)


def _rank(v):
    s = sorted(v)
    return [s.index(x) for x in v]


def _corr(a, b):
    if len(a) < 3:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


# 1) per-example view of the EVAL frames (the ones SBC is scored on) ----------
for r in res:
    d = r["mbpd_diag"]
    fam = r.get("family") or d.get("family", "lateral")
    cf_op = d.get("cf_op") or (r.get("cf_info") or {}).get("cf_op", "mirror")
    real = {p.get("probe_key"): p.get("consistent_with_lie") for p in r.get("probe_log", [])}
    print("\n=== %s / %s   [family=%s, cf_op=%s]   SBC=%.3f"
          % (r["subj"], r["obj"], fam, cf_op, r["SBC"]))
    print("  key  split  A      P(lie)  lie-side-assent  status                alpha   adv_ok  eval_probe")
    for x in d["rows"]:
        if x["split"] != "eval":
            continue
        print("  %-4s %-5s %6.2f %7.2f %10.2f       %-21s %6s   %-6s  %s"
              % (x["key"], x["split"], x["A"], x["P_signed"], x["lie_side_assent"],
                 x["status"], "-" if x["alpha"] is None else "%.2f" % x["alpha"],
                 x.get("adv_ok"), real.get(x["key"], "-")))
    print("  summary:", json.dumps(d["summary"]))

# 2) pooled hypothesis test over all 14 frames x 2 questions x examples --------
rows = [x for r in res for x in r["mbpd_diag"]["rows"] if "adv_ok" in x]
rows_h = [x for x in rows if x["status"] != "prior_already_lie"]
if not rows:
    sys.exit("\n(no adversarial gaps recorded)")
by = defaultdict(lambda: [0, 0])
by_fam = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in res:
    fam = r.get("family") or r["mbpd_diag"].get("family", "lateral")
    for x in r["mbpd_diag"]["rows"]:
        if "adv_ok" not in x:
            continue
        st = x["status"].replace("_capped", "")
        by[st][0] += int(x["adv_ok"])
        by[st][1] += 1
        by_fam[fam][st][0] += int(x["adv_ok"])
        by_fam[fam][st][1] += 1

print("\n=== POOLED over %d (frame, question) cells, %d examples" % (len(rows), len(res)))
print("  status             adv_ok / n    rate")
for st in ("prior_already_lie", "mirror_enough", "beyond_mirror", "perception_dead"):
    ok, n = by.get(st, [0, 0])
    print("  %-18s %4d / %-4d   %s" % (st, ok, n, ("%.3f" % (ok / n)) if n else "-"))

if len(by_fam) > 1:
    print("\n=== BY RELATION FAMILY (adv_ok / n)")
    for fam, fdict in sorted(by_fam.items()):
        parts = ["%s=%d/%d" % (st[:6], fdict[st][0], fdict[st][1])
                 for st in ("prior_already_lie", "mirror_enough", "beyond_mirror", "perception_dead")
                 if fdict[st][1] > 0]
        print("  %-14s %s" % (fam, "  ".join(parts)))

m = res[0]["mbpd_diag"]["margin"]
def shortfall(x):
    return m - (x["lie_side_assent"] + x["P_signed"])      # = m - s*g_cf

fails = [shortfall(x) for x in rows_h if not x["adv_ok"]]
oks = [shortfall(x) for x in rows_h if x["adv_ok"]]
avg = lambda v: sum(v) / len(v) if v else float("nan")
print("\n  (excluding prior_already_lie cells)")
print("  mean counterfactual shortfall   failed cells: %.2f (n=%d)   passed cells: %.2f (n=%d)"
      % (avg(fails), len(fails), avg(oks), len(oks)))
print("  -> hypothesis SUPPORTED if failed cells have clearly larger shortfall"
      " and perception_dead / beyond_mirror have lower adv_ok rates.")

# 3) continuous readout (pre-attack perception vs final 'No' margin) ----------
print("\n=== TRUE-question rows: does pre-attack perception predict the final margin?")
print("  example          fam       n   rho(P, margin)   rho(P_dual, margin)   rho(clean, margin)   mean margin by status")
allP, allPd, allC, allM = [], [], [], []
for r in res:
    fam = (r.get("family") or r["mbpd_diag"].get("family", "lateral"))[:8]
    by_frame = {}
    for x in r["mbpd_diag"]["rows"]:
        by_frame.setdefault(x["frame"], {})[x["q"]] = x
    xs = [x for x in r["mbpd_diag"]["rows"] if x["q"] == "true" and "g_adv" in x]
    P = [x["P_signed"] for x in xs]
    Pd = [x.get("P_dual", 0.5 * (by_frame[x["frame"]]["true"]["g_clean"]
                                 - by_frame[x["frame"]]["target"]["g_clean"])) for x in xs]
    C = [-x["g_clean"] for x in xs]
    M = [-x["g_adv"] for x in xs]
    allP += P; allPd += Pd; allC += C; allM += M
    bys = defaultdict(list)
    for x in xs:
        bys[x["status"].replace("_capped", "")].append(-x["g_adv"])
    bstr = "  ".join("%s=%.2f" % (k[:12], sum(v) / len(v)) for k, v in sorted(bys.items()))
    print("  %-15s %-8s %3d   %+.2f            %+.2f                  %+.2f               %s"
          % (r["subj"][:15], fam, len(xs), _corr(_rank(P), _rank(M)),
             _corr(_rank(Pd), _rank(M)), _corr(_rank(C), _rank(M)), bstr))
print("  %-15s %-8s %3d   %+.2f            %+.2f                  %+.2f"
      % ("POOLED", "all", len(allP), _corr(_rank(allP), _rank(allM)),
         _corr(_rank(allPd), _rank(allM)), _corr(_rank(allC), _rank(allM))))

# 4) DICD Cross-Validation & 3-Way Decomposition (on mirror-evaluated examples)
P_mir_no, P_mir_yes, P_dual_all, A_mir_no, A_dual_all, L_3way_all = [], [], [], [], [], []
for r in res:
    by_f = {}
    for x in r["mbpd_diag"]["rows"]:
        by_f.setdefault(x["frame"], {})[x["q"]] = x
    for f, d in sorted(by_f.items()):
        if "target" not in d or "true" not in d:
            continue
        gt, gr = d["target"], d["true"]
        p_dual = gt.get("P_dual", 0.5 * (gr["g_clean"] - gt["g_clean"]))
        a_dual = gt.get("A_dual", 0.5 * (gr["g_clean"] + gt["g_clean"]))
        l_3way = gt.get("L_3way", 0.25 * ((gt["g_clean"] + gt["g_mirror"]) - (gr["g_clean"] + gr["g_mirror"])))
        P_dual_all.append(p_dual)
        A_dual_all.append(a_dual)
        P_mir_no.append(gr["P_signed"])
        P_mir_yes.append(gt["P_signed"])
        A_mir_no.append(gr["A"])
        L_3way_all.append(l_3way)

if P_dual_all:
    print("\n=== DICD PROPOSITION-DUAL VALIDATION & 3-WAY BIAS DECOMPOSITION (n=%d frames)" % len(P_dual_all))
    print("  Pearson r(P_dual [clean-only], P_mirror [true-Q])   = %+.3f" % _corr(P_dual_all, P_mir_no))
    print("  Pearson r(P_dual [clean-only], P_mirror [target-Q]) = %+.3f" % _corr(P_dual_all, P_mir_yes))
    print("  Pearson r(A_dual [clean-only], A_mirror [true-Q])   = %+.3f" % _corr(A_dual_all, A_mir_no))
    print("  Mean |Lexical Word Bias L_3way|                     = %.3f logits (signed mean %+.3f)"
          % (sum(abs(x) for x in L_3way_all) / len(L_3way_all), sum(L_3way_all) / len(L_3way_all)))

evals = [(r["subj"], (r.get("family") or "lateral")[:7], x["key"], x["status"], -x["g_adv"])
         for r in res for x in r["mbpd_diag"]["rows"]
         if x["q"] == "true" and x["split"] == "eval" and "g_adv" in x]
if evals:
    print("\n  held-out 'No' frames (final margin; >0 = lie holds):")
    for e in evals:
        print("    %-15s %-7s %-4s %-19s %+.2f" % e)
    print("  mean held-out 'No' margin = %.2f   (compare arms: MBPD_LOSS should raise this)"
          % (sum(e[4] for e in evals) / len(evals)))
