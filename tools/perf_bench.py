#!/usr/bin/env python3
"""
Reading speed of the training loop, in characters per second, as one line.

    python tools/perf_bench.py                       # 20 warm-up, 200 timed
    MINAGI_COMPILE=1 python tools/perf_bench.py --warmup 40

Runs `train.py read` as it is (tools/perf_common.py) and times only the
learning steps after --warmup, so a compile, a graph capture or a cold cache
in the first steps is not counted. The clock starts as step `warmup` begins
and stops as the step after the last timed one would begin, with the card
synchronised at both ends, so it covers whole steps: forward, backward,
optimiser, and everything the loop does around them. Characters are the ones
those steps trained on.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_common import add_run_args, run_training, sync  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    add_run_args(ap, steps=200, warmup=20)
    ap.add_argument("--out", default=None,
                    help="scratch for the run (default: a temporary directory)")
    ap.add_argument("--log", default=os.devnull,
                    help="where the training loop's own output goes")
    a = ap.parse_args()

    clock = {}
    chars = [0]
    pos = {}

    def on_begin(i, reader):
        if i == a.warmup or i == a.warmup + a.steps:
            sync()
            clock[i] = time.perf_counter()
        pos[id(reader)] = reader.pos

    def on_end(i, reader, loss, info):
        if i >= a.warmup:
            chars[0] += reader.pos - pos[id(reader)]

    import tempfile
    with tempfile.TemporaryDirectory(prefix="perf-bench-") as tmp:
        done = run_training(a, a.out or tmp, on_begin=on_begin, on_end=on_end,
                            log=a.log)
    if done < a.warmup + a.steps:
        return 1
    secs = clock[a.warmup + a.steps] - clock[a.warmup]
    import torch
    env = " ".join(f"{k}={v}" for k, v in sorted(os.environ.items())
                   if k.startswith("MINAGI_"))
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"chars/s {chars[0] / secs:.1f}  steps {a.steps}  warmup "
          f"{a.warmup}  chars {chars[0]}  secs {secs:.2f}  "
          f"ms/step {1e3 * secs / a.steps:.1f}  device {dev}"
          + (f"  {env}" if env else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
