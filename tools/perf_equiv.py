#!/usr/bin/env python3
"""
Does a faster training step still compute the same thing? Record, then compare.

    python tools/perf_equiv.py --out runs/perf/eager            # 500 steps
    MINAGI_COMPILE=1 python tools/perf_equiv.py --out runs/perf/compiled
    python tools/perf_equiv.py --compare runs/perf/eager/equiv.jsonl \\
                                         runs/perf/compiled/equiv.jsonl

The run mode trains `train.py read` as it is, from a fresh model and a fixed
seed (tools/perf_common.py), and writes one JSON line per learning step to
<out>/equiv.jsonl:

  loss       the step's training loss, as the loop backpropagates it
  grad_norm  the norm clip_grad_norm_ returned for it
  depth      the recurrence depth sampled for the step (RecurCoder.sample_depth)
  ponder     the expected depth under the halting distribution
             (RecurCoder.last_steps) - the halting statistic the model keeps
  loads      experts brought onto the card by the step's forward
  admitted   the experts that forward admitted

Compare mode reports the largest loss difference, absolute and relative, and
the first step where depth, loads or admitted differ, and exits 1 when the
loss differs by more than --tol or any count differs. A run that admits one
different expert reads the rest of the run with different weights, so the
first mismatch is the one worth reading; later ones follow from it.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_common import ROOT, add_run_args, run_training  # noqa: E402

# Two eager runs on the CPU with the same seed agree to the bit
# (tests/test_perf_tools.py), so any difference at all is a real one there.
# A compiled or CUDA-graphed step reorders floating-point work and will not
# match eager to the bit; pass the --tol that run is judged at.
DEFAULT_TOL = 0.0


def record(a):
    import torch
    rows = []

    def on_end(i, reader, loss, info):
        p = info["ponder"]
        rows.append({"step": i, "loss": float(loss.detach()),
                     "grad_norm": None, **info,
                     "ponder": None if p is None else float(p)})

    clip = torch.nn.utils.clip_grad_norm_

    def clip_grad_norm_(*args, **kw):
        n = clip(*args, **kw)
        if rows and rows[-1]["grad_norm"] is None:
            rows[-1]["grad_norm"] = float(n)
        return n

    done = run_training(a, a.out, on_end=on_end,
                        patches=[(torch.nn.utils, "clip_grad_norm_",
                                  clip_grad_norm_)],
                        log=os.path.join(a.out, "train.log"))
    path = os.path.join(a.out, "equiv.jsonl")
    meta = {"config": os.path.relpath(os.path.abspath(a.config), ROOT),
            "data": a.data, "seed": a.seed, "steps": done,
            "device": a.device or ("cuda" if torch.cuda.is_available()
                                   else "cpu"),
            "torch": torch.__version__,
            "env": {k: v for k, v in os.environ.items()
                    if k.startswith("MINAGI_")},
            "train_args": a.train_arg}
    with open(path, "w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{done} steps -> {path}")
    return 0 if done == a.warmup + a.steps else 1


def load(path):
    meta, rows = {}, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "meta" in r:
                meta = r["meta"]
            else:
                rows.append(r)
    return meta, rows


def compare(pa, pb, tol, ponder_tol=None):
    """The differences between two recordings, as a dict; `ok` says whether
    they agree within `tol` on loss (and ponder) and exactly on the counts."""
    ponder_tol = tol if ponder_tol is None else ponder_tol
    ma, ra = load(pa)
    mb, rb = load(pb)
    n = min(len(ra), len(rb))
    res = {"steps": [len(ra), len(rb)], "max_abs_loss": 0.0,
           "max_rel_loss": 0.0, "max_abs_loss_step": None,
           "max_abs_grad_norm": 0.0, "max_abs_ponder": 0.0,
           "first_count_mismatch": None, "count_mismatches": 0}
    for i in range(n):
        x, y = ra[i], rb[i]
        d = abs(x["loss"] - y["loss"])
        if d > res["max_abs_loss"]:
            res["max_abs_loss"], res["max_abs_loss_step"] = d, x["step"]
        res["max_rel_loss"] = max(res["max_rel_loss"],
                                  d / max(abs(x["loss"]), 1e-12))
        if x.get("grad_norm") is not None and y.get("grad_norm") is not None:
            res["max_abs_grad_norm"] = max(res["max_abs_grad_norm"],
                                           abs(x["grad_norm"] - y["grad_norm"]))
        if x.get("ponder") is not None and y.get("ponder") is not None:
            res["max_abs_ponder"] = max(res["max_abs_ponder"],
                                        abs(x["ponder"] - y["ponder"]))
        bad = [k for k in ("depth", "loads", "admitted") if x.get(k) != y.get(k)]
        if bad:
            res["count_mismatches"] += 1
            if res["first_count_mismatch"] is None:
                res["first_count_mismatch"] = {
                    "step": x["step"], "fields": bad,
                    "a": {k: x.get(k) for k in bad},
                    "b": {k: y.get(k) for k in bad}}
    res["ok"] = (len(ra) == len(rb) and n > 0
                 and res["max_abs_loss"] <= tol
                 and res["max_abs_ponder"] <= ponder_tol
                 and res["count_mismatches"] == 0)
    res["meta"] = [ma, mb]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_run_args(ap, steps=500)
    ap.add_argument("--out", default="runs/perf/equiv",
                    help="where equiv.jsonl is written")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two equiv.jsonl recordings instead of running")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="largest absolute loss difference that still agrees")
    ap.add_argument("--ponder-tol", type=float, default=None,
                    help="the same for the expected depth (default: --tol)")
    a = ap.parse_args()
    if not a.compare:
        return record(a)
    r = compare(*a.compare, a.tol, a.ponder_tol)
    print(f"steps        {r['steps'][0]} vs {r['steps'][1]}")
    print(f"loss         max |diff| {r['max_abs_loss']:.3e} at step "
          f"{r['max_abs_loss_step']}, max rel {r['max_rel_loss']:.3e} "
          f"(tol {a.tol:g})")
    print(f"grad norm    max |diff| {r['max_abs_grad_norm']:.3e}")
    print(f"ponder       max |diff| {r['max_abs_ponder']:.3e}")
    fm = r["first_count_mismatch"]
    if fm:
        print(f"counts       {r['count_mismatches']} step(s) differ; first at "
              f"step {fm['step']}: {', '.join(fm['fields'])}  "
              f"a={fm['a']}  b={fm['b']}")
    else:
        print("counts       depth, loads and admitted agree at every step")
    print("EQUAL" if r["ok"] else "DIFFERENT")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
