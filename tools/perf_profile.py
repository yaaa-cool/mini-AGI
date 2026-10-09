#!/usr/bin/env python3
"""
Where a training step's wall time goes, under torch.profiler.

    python tools/perf_profile.py --out runs/perf/profile     # 5 warm-up, 30 profiled
    python tools/perf_trace_summary.py runs/perf/profile/trace.json.gz
    python tools/perf_profile.py --steps 10 --table --no-trace

Runs `train.py read` as it is (tools/perf_common.py), lets --warmup steps go
by unprofiled and --prof-warmup more under a warming profiler, then records
--steps whole learning steps (CPU, and CUDA when there is a card) in one
profiler cycle and writes to --out:

  trace.json.gz the chrome trace, gzipped (ui.perfetto.dev opens it as is);
                tools/perf_trace_summary.py reads it into the numbers below
  profile.txt   with --table only: the breakdown below, then torch's own
                tables, the top ops by self CPU time and, on a card, by self
                device time. torch builds them in Python, single-threaded:
                minutes for a few steps of this model on a card
  train.log     what the training loop printed

The run marks two kinds of range with record_function, so they show in the
trace: the expert paging code (Tiers fetch / disk / writeback, PagedPool
admit / place / rearrange / flush, read_npz) as paging:<what>, and the expert
dispatch the pool runs under gradient checkpointing (PooledMLP's run_exact or
padded run) as pool:experts - in the forward, and again where the backward
recomputes it.

The breakdown, as fractions of the profiled wall time:

  in torch ops       host time inside any profiled op, on any thread (the
                     union, so the autograd thread's backward is counted once)
  python / other     wall time with no op running on the host: the Python
                     around the ops, and anything not instrumented
  syncs              host reads of a device value (aten::_local_scalar_dense
                     is every .item() / float() / int() / bool() of a tensor)
                     and the CUDA calls that wait: cudaStreamSynchronize,
                     cudaDeviceSynchronize, cudaMemcpy*
  kernel launches    cudaLaunchKernel and kin: how many per step, host time
  gpu busy           the union of kernel and memcpy time on the card
  paging             time inside the paging ranges; nested scopes are listed
                     separately, the total counts only the outermost
"""

import argparse
import functools
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_common import add_run_args, run_training, sync  # noqa: E402

SYNC_OPS = ("aten::_local_scalar_dense", "aten::item")
SYNC_CUDA = ("cudaStreamSynchronize", "cudaDeviceSynchronize",
             "cudaEventSynchronize", "cudaMemcpy", "cudaMemcpyAsync",
             "cudaMemcpy2DAsync")
LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel",
          "cuLaunchKernelEx", "hipLaunchKernel", "hipExtModuleLaunchKernel")


def paging_patches(acc, active):
    """(object, name, wrapper) for every paging entry point, each timing
    itself into `acc` while `active[0]` and marking a record_function."""
    import torch
    import minagi.paged as paged

    def wrap(fn, label):
        @functools.wraps(fn)
        def w(*args, **kw):
            if not active[0]:
                return fn(*args, **kw)
            t = time.perf_counter()
            with torch.profiler.record_function(label):
                try:
                    return fn(*args, **kw)
                finally:
                    a = acc[label]
                    a[0] += 1
                    a[1] += time.perf_counter() - t
        return w

    out = []
    for cls, names in ((paged.Tiers, ("fetch", "_from_disk", "_to_disk",
                                      "flush")),
                       (paged.PagedPool, ("admit", "_place", "_rearrange",
                                          "flush"))):
        for n in names:
            out.append((cls, n, wrap(getattr(cls, n),
                                     f"paging:{cls.__name__}.{n}")))
    out.append((paged, "read_npz", wrap(paged.read_npz, "paging:read_npz")))
    return out


def _union(spans):
    """Total length covered by a list of (start, end)."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(spans):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def cycle_stats(evs):
    """The breakdown's sums over one profiler cycle's events. The wall is the
    cycle's ProfilerStep spans, so the analysis between cycles is not in it."""
    from torch.autograd import DeviceType
    host = [e for e in evs if e.device_type == DeviceType.CPU]
    # kernels and copies only: a record_function also shows on the card's
    # timeline (a user annotation), spanning everything queued inside it
    dev = [e for e in evs if e.device_type != DeviceType.CPU
           and not getattr(e, "is_user_annotation", False)]

    def outermost(e):
        p = e.cpu_parent
        return p is None or p.name.startswith("ProfilerStep")

    def span(e):
        return (e.time_range.start, e.time_range.end)

    # each step is one ProfilerStep event (named "ProfilerStep*" once torch
    # has gathered them)
    steps = [e.time_range.elapsed_us() for e in host
             if e.name.startswith("ProfilerStep")]
    top = [e for e in host if not e.name.startswith("ProfilerStep")
           and outermost(e)]
    n = defaultdict(int)
    t = defaultdict(float)
    sync_spans = []
    for e in host:
        if e.name in SYNC_OPS + SYNC_CUDA + LAUNCH:
            n[e.name] += 1
            t[e.name] += e.time_range.elapsed_us()
            if e.name not in LAUNCH:
                sync_spans.append(span(e))

    def paging_root(e):
        p = e.cpu_parent
        while p is not None:
            if p.name.startswith("paging:"):
                return False
            p = p.cpu_parent
        return True

    return {"wall": sum(steps), "steps": len(steps),
            "in_ops": _union([span(e) for e in top]), "top": len(top),
            "sync": _union(sync_spans), "n": n, "t": t, "dev": len(dev),
            "paging": sum(e.time_range.elapsed_us() for e in host
                          if e.name.startswith("paging:") and paging_root(e)),
            "gpu": _union([span(e) for e in dev])}


def add_stats(tot, s):
    for k, v in s.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                tot[k][kk] += vv
        else:
            tot[k] += v


def breakdown(tot, steps, acc):
    wall_us, in_ops, top = tot["wall"], tot["in_ops"], tot["top"]
    n, t, ts, gpu_us = tot["n"], tot["t"], tot["sync"], tot["gpu"]
    paging_us, dev = tot["paging"], tot["dev"]

    def pct(us):
        return f"{100.0 * us / max(wall_us, 1e-9):5.1f}%"

    def ms(us):
        return f"{us / 1e3:10.1f} ms"

    L = []
    L.append(f"wall            {ms(wall_us)}  over {steps} steps, "
             f"{wall_us / 1e3 / max(steps, 1):.1f} ms/step")
    L.append(f"in torch ops    {ms(in_ops)}  {pct(in_ops)}  "
             f"({top / max(steps, 1):.0f} outermost ops/step)")
    L.append(f"python / other  {ms(wall_us - in_ops)}  "
             f"{pct(wall_us - in_ops)}  host time with no op running")
    # an .item() on a card is a _local_scalar_dense holding a memcpy and a
    # stream sync, so the time is the union and the counts are per kind
    L.append(f"syncs           {ms(ts)}  {pct(ts)}  "
             f"({n['aten::_local_scalar_dense'] / max(steps, 1):.1f} host "
             f"reads of a tensor value/step)")
    for k in SYNC_OPS + SYNC_CUDA:
        if n[k]:
            L.append(f"  {k:<28} {n[k]:7d} calls  {n[k] / max(steps, 1):7.1f}"
                     f"/step  {t[k] / 1e3:9.1f} ms")
    nl = sum(n[k] for k in LAUNCH)
    tl = sum(t[k] for k in LAUNCH)
    L.append(f"kernel launches {ms(tl)}  {pct(tl)}  "
             f"({nl} launches, {nl / max(steps, 1):.0f}/step)"
             + ("" if nl else "  - none: no card in this run"))
    L.append(f"gpu busy        {ms(gpu_us)}  {pct(gpu_us)}"
             + ("" if dev else "  - no device activity recorded"))
    L.append(f"paging          {ms(paging_us)}  {pct(paging_us)}  "
             f"(outermost paging scopes)")
    for label in sorted(acc):
        c, s = acc[label]
        L.append(f"  {label:<28} {c:7d} calls  {c / max(steps, 1):7.1f}"
                 f"/step  {s * 1e3:9.1f} ms")
    return "\n".join(L)


def pool_patches():
    """Mark the expert dispatch the pool checkpoints as pool:experts. The
    wrapped function is what the backward recomputes, so the recompute is
    marked too."""
    import torch
    import minagi.pool as pool
    ckpt = pool.checkpoint

    def checkpoint(fn, *args, **kw):
        def experts(*a):
            with torch.profiler.record_function("pool:experts"):
                return fn(*a)
        return ckpt(experts, *args, **kw)
    return [(pool, "checkpoint", checkpoint)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_run_args(ap, steps=30, warmup=5)
    ap.add_argument("--prof-warmup", type=int, default=3,
                    help="steps under the profiler before it records")
    ap.add_argument("--out", default="runs/perf/profile")
    ap.add_argument("--no-trace", action="store_true",
                    help="skip the chrome trace")
    ap.add_argument("--table", action="store_true",
                    help="also write profile.txt from torch's own event "
                         "lists (slow: see above)")
    ap.add_argument("--rows", type=int, default=40,
                    help="rows in each of torch's tables")
    ap.add_argument("--shapes", action="store_true",
                    help="record input shapes (record_shapes)")
    ap.add_argument("--stack", action="store_true",
                    help="record Python stacks (with_stack)")
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile, schedule
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and (a.device or "cuda").startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)
    trace = None if a.no_trace else os.path.join(a.out, "trace.json.gz")
    acc = defaultdict(lambda: [0, 0.0])
    done_at = [None]

    def on_trace(p):
        if trace is not None:
            p.export_chrome_trace(trace)
            print(f"-> {trace}")
        if not a.table:
            return
        tot = {"n": defaultdict(int), "t": defaultdict(float)}
        tot.update((k, 0) for k in ("wall", "steps", "in_ops", "top", "sync",
                                    "dev", "paging", "gpu"))
        add_stats(tot, cycle_stats(list(p.events())))
        text = breakdown(tot, a.steps, acc)
        print(text)
        avg = p.key_averages()
        parts = [text, "", "# top ops by self CPU time",
                 avg.table(sort_by="self_cpu_time_total", row_limit=a.rows)]
        if ProfilerActivity.CUDA in acts:
            for key in ("self_device_time_total", "self_cuda_time_total"):
                try:
                    tab = avg.table(sort_by=key, row_limit=a.rows)
                except (AttributeError, KeyError, ValueError):
                    continue
                parts += ["", "# top ops by self device time", tab]
                break
        path = os.path.join(a.out, "profile.txt")
        with open(path, "w") as f:
            f.write("\n".join(parts) + "\n")
        print(f"-> {path}")

    prof = profile(activities=acts, on_trace_ready=on_trace,
                   record_shapes=a.shapes, with_stack=a.stack,
                   schedule=schedule(wait=0, warmup=a.prof_warmup,
                                     active=a.steps, repeat=1))
    active = [False]
    start = a.warmup
    record = start + a.prof_warmup
    end = record + a.steps

    def on_begin(i, reader):
        if i == start:
            sync()
            prof.start()
        elif i == end:
            sync()
            active[0] = False
            prof.stop()
            done_at[0] = i
        elif start < i < end:
            prof.step()
        if i == record:
            active[0] = True

    os.makedirs(a.out, exist_ok=True)
    ra = argparse.Namespace(**vars(a))
    ra.warmup = record                    # run_training stops at warmup + steps
    done = run_training(ra, a.out, on_begin=on_begin,
                        patches=paging_patches(acc, active) + pool_patches(),
                        log=os.path.join(a.out, "train.log"))
    if done < end or done_at[0] is None:
        if start < done:
            prof.stop()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
