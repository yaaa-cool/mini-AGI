#!/usr/bin/env python3
"""
Per-step numbers from a torch.profiler chrome trace, without torch.

    python tools/perf_trace_summary.py runs/perf/profile/trace.json.gz
    python tools/perf_trace_summary.py trace.json.gz --out profile.txt --top 20

Reads the trace perf_profile.py writes (gzipped or not; streamed with ijson
when it is installed, else loaded whole) in one pass and prints, per profiled
step (a ProfilerStep range):

  wall / ops         the steps' wall time; host time inside any torch op (the
                     union over threads of the outermost ops) and the rest,
                     Python and anything not instrumented
  syncs              host reads of a tensor (aten::_local_scalar_dense), the
                     waiting runtime calls (cuda*Synchronize, cudaMemcpy) and
                     device-to-host copies, with their host time
  launches           cudaLaunchKernel and kin, and every other runtime call
  gpu                kernels, copies and sets: count, GPU time, the union
                     (busy) as a share of the wall
  regions            host and GPU time of the record_function ranges in the
                     trace - pool:experts (forward; recompute, where the
                     backward runs it again; and the backward of the ops it
                     ran, matched by autograd sequence number), paging:*,
                     the optimiser - and of the rest, forward and backward
  top ops            by self host time, and by the GPU time of the kernels
                     each launched (the innermost op around the launch)
  top kernels        by GPU time
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict

SYNC_RT = ("cudaStreamSynchronize", "cudaDeviceSynchronize",
           "cudaEventSynchronize", "cudaMemcpy", "hipStreamSynchronize",
           "hipDeviceSynchronize", "hipMemcpy")
COPY_RT = ("cudaMemcpyAsync", "cudaMemcpy2DAsync", "hipMemcpyAsync")
LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel",
          "cuLaunchKernelEx", "hipLaunchKernel", "hipExtModuleLaunchKernel")
HOST_READ = "aten::_local_scalar_dense"
BWD = "autograd::engine::evaluate_function: "
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
RT_CATS = ("cuda_runtime", "cuda_driver")


def events(path):
    opener = gzip.open if path.endswith(".gz") else open
    f = opener(path, "rb")
    try:
        import ijson
    except ImportError:
        with f:
            yield from json.load(f)["traceEvents"]
        return
    with f:
        yield from ijson.items(f, "traceEvents.item", use_float=True)


def union(spans):
    tot, cs, ce = 0.0, None, None
    for s, e in sorted(spans):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def region(name):
    if name.startswith("Optimizer."):
        return "optimizer"
    if name.startswith("paging:"):
        return "paging"
    return name


def summarise(path, top=10):
    ops, anns, rts, gpu = [], [], [], []
    for e in events(path):
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        ts, dur = float(e["ts"]), float(e.get("dur", 0.0))
        if cat == "cpu_op":
            seq = (e.get("args") or {}).get("Sequence number")
            ops.append((e["tid"], ts, dur, e["name"], seq))
        elif cat == "user_annotation":
            anns.append((e["tid"], ts, dur, e["name"]))
        elif cat in RT_CATS:
            rts.append((e["tid"], ts, dur, e["name"],
                        (e.get("args") or {}).get("correlation")))
        elif cat in GPU_CATS:
            gpu.append((ts, dur, e["name"], cat,
                        (e.get("args") or {}).get("correlation")))

    steps = [a for a in anns if a[3].startswith("ProfilerStep")]
    n = max(len(steps), 1)
    wall = sum(a[2] for a in steps)
    main = steps[0][0] if steps else None

    # one walk per thread over ops, ranges and runtime calls, nested by time:
    # each item learns the outermost range it is in, the innermost op around
    # it, and the sequence number of the backward node it runs under
    items = defaultdict(list)
    for i, o in enumerate(ops):
        items[o[0]].append((o[1], -o[2], 0, i))
    for i, a in enumerate(anns):
        if not a[3].startswith("ProfilerStep"):
            items[a[0]].append((a[1], -a[2], 1, i))
    for i, r in enumerate(rts):
        items[r[0]].append((r[1], -r[2], 2, i))
    op_self = [o[2] for o in ops]
    op_ctx = [None] * len(ops)          # (region, bwd seq) the op is in
    outer = []                          # spans of the outermost ops
    rt_ctx = [None] * len(rts)          # (region, innermost op, bwd seq, tid)
    ann_outer = []                      # outermost ranges: (tid, index)
    op_outer = []                       # outermost ops outside every range
    for tid, its in items.items():
        its.sort()
        stack = []                      # [end, region, op index, bwd seq]
        for ts, negdur, kind, i in its:
            end = ts - negdur
            while stack and stack[-1][0] <= ts:
                stack.pop()
            reg, op, bseq = stack[-1][1:] if stack else (None, None, None)
            if kind == 0:
                if op is not None:
                    op_self[op] -= ops[i][2]
                else:
                    outer.append((ts, end))
                    if reg is None:
                        op_outer.append(i)
                op_ctx[i] = (reg, bseq)
                name, seq = ops[i][3], ops[i][4]
                if bseq is None and name.startswith(BWD):
                    bseq = seq
                stack.append([end, reg, i, bseq])
            elif kind == 1:
                if reg is None:
                    ann_outer.append(i)
                    reg = region(anns[i][3])
                stack.append([end, reg, op, bseq])
            else:
                rt_ctx[i] = (reg, op, bseq, tid)

    # forward ops of the expert dispatch, by the sequence numbers their
    # backward nodes carry
    expert_seq = {ops[i][4] for i in range(len(ops))
                  if ops[i][0] == main and op_ctx[i][0] == "pool:experts"
                  and ops[i][4] not in (None, -1)}

    def where(reg, bseq, tid):
        if reg == "pool:experts":
            return "pool:experts fwd" if tid == main else \
                "pool:experts recompute"
        if reg is not None:
            return reg
        if bseq is not None and bseq in expert_seq:
            return "pool:experts bwd"
        return "rest fwd" if tid == main else "rest bwd"

    by_corr = {}
    for i, r in enumerate(rts):
        if r[4] is not None:
            by_corr[r[4]] = i

    # GPU time: per region, per launching op, per kernel
    g_reg = defaultdict(lambda: [0, 0.0])
    g_op = defaultdict(float)
    g_kern = defaultdict(lambda: [0, 0.0])
    g_cat = defaultdict(lambda: [0, 0.0])
    d2h = [0, 0.0]
    d2h_corr = set()
    for ts, dur, name, cat, corr in gpu:
        g_cat[cat][0] += 1
        g_cat[cat][1] += dur
        g_kern[name][0] += 1
        g_kern[name][1] += dur
        if cat == "gpu_memcpy" and "DtoH" in name:
            d2h[0] += 1
            d2h[1] += dur
            d2h_corr.add(corr)
        ri = by_corr.get(corr)
        if ri is None:
            g_reg["(unattributed)"][0] += 1
            g_reg["(unattributed)"][1] += dur
            continue
        reg, op, bseq, tid = rt_ctx[ri]
        k = where(reg, bseq, tid)
        g_reg[k][0] += 1
        g_reg[k][1] += dur
        g_op[ops[op][3] if op is not None else rts[ri][3]] += dur
    busy = union((ts, ts + dur) for ts, dur, *_ in gpu)

    # host time per region: outermost ranges, and outermost ops elsewhere
    h_reg = defaultdict(float)
    for i in ann_outer:
        tid, ts, dur, name = anns[i]
        k = region(name)
        if k == "pool:experts":
            k = where(k, None, tid)
        h_reg[k] += dur
    for i in op_outer:
        tid, ts, dur, name, seq = ops[i]
        h_reg[where(None, seq if name.startswith(BWD) else None, tid)] += dur
    paging = defaultdict(lambda: [0, 0.0])
    for a in anns:
        if a[3].startswith("paging:"):
            paging[a[3]][0] += 1
            paging[a[3]][1] += a[2]

    # syncs and launches
    rt_n = defaultdict(int)
    rt_t = defaultdict(float)
    sync_spans = []
    for i, r in enumerate(rts):
        rt_n[r[3]] += 1
        rt_t[r[3]] += r[2]
        if r[3] in SYNC_RT or (r[3] in COPY_RT and r[4] in d2h_corr):
            sync_spans.append((r[1], r[1] + r[2]))
    reads = sum(1 for o in ops if o[3] == HOST_READ)
    in_ops = union(outer)
    self_t = defaultdict(lambda: [0, 0.0])
    for i, o in enumerate(ops):
        self_t[o[3]][0] += 1
        self_t[o[3]][1] += op_self[i]

    def ms(us):
        return f"{us / 1e3 / n:9.2f} ms/step"

    def pct(us):
        return f"{100.0 * us / max(wall, 1e-9):5.1f}%"

    L = [f"trace {path}",
         f"steps           {len(steps)}  (ProfilerStep ranges)",
         f"wall            {ms(wall)}",
         f"in torch ops    {ms(in_ops)}  {pct(in_ops)}  "
         f"({len(outer) / n:.0f} outermost ops/step, {len(ops) / n:.0f} "
         f"ops/step in all)",
         f"python / other  {ms(wall - in_ops)}  {pct(wall - in_ops)}",
         f"syncs           {ms(union(sync_spans))}  "
         f"{pct(union(sync_spans))}  ({reads / n:.1f} host reads of a "
         f"tensor/step, {d2h[0] / n:.1f} device-to-host copies/step, "
         f"{d2h[1] / 1e3 / n:.2f} ms GPU)"]
    for k in SYNC_RT:
        if rt_n[k]:
            L.append(f"  {k:<34} {rt_n[k] / n:8.1f}/step {ms(rt_t[k])}")
    nl = sum(rt_n[k] for k in LAUNCH)
    tl = sum(rt_t[k] for k in LAUNCH)
    L.append(f"kernel launches {ms(tl)}  {pct(tl)}  ({nl / n:.0f}/step)")
    L.append("runtime calls (all)")
    for k in sorted((k for k in rt_t if rt_n[k]), key=rt_t.get,
                    reverse=True)[:8]:
        L.append(f"  {k:<34} {rt_n[k] / n:8.1f}/step {ms(rt_t[k])}")
    L.append(f"gpu busy        {ms(busy)}  {pct(busy)}")
    for k in GPU_CATS:
        if g_cat[k][0]:
            L.append(f"  {k:<34} {g_cat[k][0] / n:8.1f}/step "
                     f"{ms(g_cat[k][1])}")
    L.append("regions                        host            gpu       "
             "kernels/step")
    for k in sorted(set(g_reg) | set(h_reg),
                    key=lambda k: -g_reg[k][1] if k in g_reg else 0):
        h = f"{h_reg[k] / 1e3 / n:9.2f} ms" if k in h_reg else " " * 12
        c, t = g_reg[k] if k in g_reg else (0, 0.0)
        L.append(f"  {k:<26} {h}  {t / 1e3 / n:9.2f} ms  {c / n:9.1f}")
    L.append("  (host: the outermost ranges and ops; a backward node that "
             "recomputes the experts holds the recompute's host time too)")
    if paging:
        L.append("paging ranges (all, nested included)")
        for k in sorted(paging):
            c, t = paging[k]
            L.append(f"  {k:<34} {c / n:8.1f}/step {ms(t)}")
    L.append(f"\ntop {top} ops by self host time")
    for k in sorted(self_t, key=lambda k: -self_t[k][1])[:top]:
        c, t = self_t[k]
        L.append(f"  {k[:60]:<60} {c / n:8.1f}/step {ms(t)}")
    L.append(f"\ntop {top} ops by GPU time of the kernels they launched")
    for k in sorted(g_op, key=g_op.get, reverse=True)[:top]:
        L.append(f"  {k[:60]:<60} {ms(g_op[k])}")
    L.append(f"\ntop {top} kernels by GPU time")
    for k in sorted(g_kern, key=lambda k: -g_kern[k][1])[:top]:
        c, t = g_kern[k]
        L.append(f"  {k[:110]:<110} {c / n:8.1f}/step {ms(t)}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--out", default=None, help="also write the summary here")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    text = summarise(a.trace, a.top)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
