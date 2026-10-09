"""
Driving `train.py read` for a fixed number of steps, for the perf tools.

perf_profile.py, perf_equiv.py and perf_bench.py all need the same thing:
the real training loop, as `train.py read` runs it, stopped after exactly N
learning steps, with a hook at the start and end of every step. Nothing in
train.py or minagi/ is edited for it. The loop's one per-step call,
FileReader.step, is wrapped for the length of the run, and a step that would
start past the last one raises out of the loop.

The config is chosen by pointing minagi.config.DEFAULT at it before the
argument parser reads its defaults, so the run is configured exactly as
`cp config.small.yaml config.yaml && python train.py read ...` would be,
without touching config.yaml. Whatever the environment says (MINAGI_COMPILE
and the like) is left to the code that reads it.

The model is made fresh from that config in a temporary directory unless
--weights-dir names one, and the read is a dry read (no --save), so nothing
outside --out is written.
"""

import contextlib
import gc
import os
import random
import signal
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _Stop(Exception):
    """Raised at the start of the step after the last one wanted."""


def add_run_args(ap, steps, warmup=0):
    ap.add_argument("--config", default=os.path.join(ROOT, "config.small.yaml"),
                    help="the config the run is built and trained from")
    ap.add_argument("--data", nargs="+", default=None,
                    help="what to read (default: data.train from the config)")
    ap.add_argument("--steps", type=int, default=steps,
                    help="learning steps measured")
    ap.add_argument("--warmup", type=int, default=warmup,
                    help="learning steps run first and not measured")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="default: cuda when available, else cpu")
    ap.add_argument("--weights-dir", default=None,
                    help="an existing weights directory to start from, read "
                         "dry (never written). Default: a fresh model from "
                         "--config in a temporary directory")
    ap.add_argument("--train-arg", action="append", default=[],
                    help="extra argument passed to `train.py read`, e.g. "
                         "--train-arg=--precision=fp32 (repeatable)")


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield old
    finally:
        setattr(obj, name, old)


def run_training(a, out, on_begin=None, on_end=None, patches=(), log=None):
    """
    Run `train.py read` until a.warmup + a.steps learning steps have finished.

    on_begin(i, reader) is called as learning step i (0-based) starts and once
    more with i == a.warmup + a.steps just before the run is stopped.
    on_end(i, reader, loss, info) is called after step i's forward, with
    `info` holding what the forward exposed: `loads` (experts brought onto
    the card), `admitted` (the experts this forward admitted), `depth` (the
    recurrence depth sampled for it) and `ponder` (the expected depth under
    the halting distribution, as the device tensor RecurCoder.last_steps
    reads back - so the step waits for it only if on_end reads it).
    `patches` is a list of (object, attribute, value) applied for the length
    of the run. `log`, when given, is the file the training loop's own
    printing goes to.

    Returns the number of learning steps finished.
    """
    import numpy as np
    import torch
    import minagi.config as mcfg

    os.makedirs(out, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    total = a.warmup + a.steps
    old_default = mcfg.DEFAULT
    mcfg.DEFAULT = os.path.abspath(a.config)
    try:
        import train
        from minagi.recur import RecurCoder
        from minagi.stream import FileReader
        c = mcfg.load()
        data = a.data or [mcfg.get(c, "data.train", "data/train")]
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

        tmp = None
        wdir = a.weights_dir
        if wdir is None:
            tmp = tempfile.TemporaryDirectory(prefix="perf-weights-", dir=out)
            wdir = os.path.join(tmp.name, "weights")
        argv = ["train.py", "--device", device, "read", *data,
                "--weights-dir", wdir, "--seed", str(a.seed),
                "--held-out", "", "--no-plots",
                "--history", os.path.join(out, "history.jsonl"),
                "--sample-log", os.path.join(out, "samples.txt"),
                *a.train_arg]

        done = [0]
        depth = [None]
        orig_step = FileReader.step
        orig_depth = RecurCoder.sample_depth

        def sample_depth(self):
            n = orig_depth(self)
            depth[0] = n
            return n

        def step(self, learn=True, aux_weight=0.0):
            if not learn:
                return orig_step(self, learn, aux_weight)
            i = done[0]
            if on_begin is not None:
                on_begin(i, self)
            if i >= total:
                raise _Stop
            pool = getattr(self.model, "pool", None)
            loads0 = getattr(pool, "loads", 0)
            depth[0] = None
            loss = orig_step(self, learn, aux_weight)
            if loss is None:
                return None
            done[0] += 1
            if on_end is not None:
                on_end(i, self, loss, {
                    "loads": int(getattr(pool, "loads", 0) - loads0),
                    "admitted": sorted(int(e) for e in
                                       getattr(pool, "_admitted", ()) or ()),
                    "depth": depth[0],
                    "ponder": getattr(self.model, "_last_steps", None)})
            return loss

        handlers = {s: signal.getsignal(s)
                    for s in (signal.SIGINT, signal.SIGTERM)}
        with contextlib.ExitStack() as st:
            st.enter_context(_patched(FileReader, "step", step))
            st.enter_context(_patched(RecurCoder, "sample_depth", sample_depth))
            for obj, name, value in patches:
                st.enter_context(_patched(obj, name, value))
            st.enter_context(_patched(sys, "argv", argv))
            if log is not None:
                # what the training loop prints, out of the tool's own output
                st.enter_context(contextlib.redirect_stdout(
                    st.enter_context(open(log, "w"))))
            try:
                train.main()
            except _Stop:
                pass
            except SystemExit as e:
                if e.code not in (0, None):
                    raise
            finally:
                for s, h in handlers.items():
                    signal.signal(s, h)
                if tmp is not None:
                    gc.collect()        # the pool's dry overlay lives in tmp
                    tmp.cleanup()
        if done[0] < total:
            print(f"[perf] the read ended after {done[0]} of {total} steps "
                  f"- give it more data", file=sys.stderr)
        return done[0]
    finally:
        mcfg.DEFAULT = old_default


def sync():
    """Wait for the card, so a timer reads the work and not its queueing."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
