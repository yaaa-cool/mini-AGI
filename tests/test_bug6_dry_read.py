"""#6: a read without --save must leave the weights directory byte-identical."""

import hashlib
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from minagi.create import create  # noqa: E402


def _digest(path):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            f = os.path.join(root, name)
            h.update(os.path.relpath(f, path).encode())
            with open(f, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def _tiny(path):
    create(path, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=64, max_steps=1, top_k=1)


def test_evicted_dirty_expert_goes_to_overlay(tmp_path):
    from minagi.dryread import overlay
    src = str(tmp_path / "weights")
    _tiny(src)
    before = _digest(src)
    _, _, pool, _ = train.build_paged(src, torch.device("cpu"), resident=1,
                                      ram_capacity=1, ceiling=64)
    overlay(pool, src)
    tiers = pool.tiers
    changed = {k: v + 1.0 for k, v in tiers.fetch(0).items()}
    tiers.put(0, changed)                   # trained while resident: dirty
    tiers.fetch(1)                          # a cache of one evicts expert 0
    tiers.fetch(2)
    assert tiers.writebacks >= 1
    # what it learned survives the eviction ...
    back = tiers.fetch(0)
    for k in ("w1", "w3", "w2"):
        assert torch.equal(back[k], changed[k]), k
    # ... and the source never saw it
    assert os.path.exists(os.path.join(pool._dry_overlay.name, "e00000.npz"))
    assert _digest(src) == before


def test_read_without_save_leaves_weights_untouched(tmp_path):
    src = str(tmp_path / "weights")
    _tiny(src)
    text = tmp_path / "input.txt"
    text.write_text("".join(chr(97 + (i * 7) % 26) for i in range(4096)))
    before = _digest(src)
    proc = subprocess.run(
        [sys.executable, "train.py", "--device", "cpu", "read", str(text),
         "--weights-dir", src, "--chunk", "8", "--context-start", "64",
         "--context-end", "64", "--resident", "1", "--ram-capacity", "1",
         "--passage", "64", "--grow-k", "0", "--held-out", "",
         "--history", "", "--sample-log", str(tmp_path / "samples.txt"),
         "--precision", "fp32", "--train-steps-mean", "0",
         "--save-every", "0", "--no-plots"],
        cwd=ROOT, text=True, capture_output=True, timeout=300,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert _digest(src) == before
    # and the overlay went with the process
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".minagi-dry-")]
