"""The perf harness (tools/perf_*.py) runs end to end on the CPU, and two eager
runs from the same seed record the same steps - to the bit, which is what
perf_equiv.DEFAULT_TOL assumes."""

import json
import os
import subprocess
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")


def tiny(tmp_path):
    """config.small.yaml shrunk to seconds, and a corpus of three subjects."""
    with open(os.path.join(ROOT, "config.small.yaml")) as f:
        c = yaml.safe_load(f)
    c["model"].update(d_model=32, n_head=2, d_ff=64, n_prelude=1, max_steps=4,
                      train_steps_mean=2.0, bptt_window=4,
                      context_start=256, context_end=256)
    c["pool"].update(experts=8, width=32, resident=4, ram_cache=6, top_k=2)
    c["training"].update(chunk=64, precision="fp32")
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump(c))
    rng = np.random.default_rng(1)
    words = ("the cat sat on a mat and then ran far def f(x): return "
             "chess e4 e5 story of old").split()
    for s in range(3):
        d = tmp_path / "corpus" / f"subj{s}"
        d.mkdir(parents=True)
        (d / "f.txt").write_text(" ".join(rng.choice(words, 600)))
    return str(cfg), str(tmp_path / "corpus")


def run(tool, *args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, os.path.join(TOOLS, tool), *args],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=600)
    assert p.returncode in (0, 1), p.stdout[-2000:] + p.stderr[-4000:]
    return p


def test_equiv_two_eager_runs_agree(tmp_path):
    cfg, corpus = tiny(tmp_path)
    common = ["--config", cfg, "--data", corpus, "--steps", "12",
              "--device", "cpu"]
    for name in ("a", "b"):
        p = run("perf_equiv.py", *common, "--out", str(tmp_path / name))
        assert p.returncode == 0, p.stdout + p.stderr
    a, b = (str(tmp_path / n / "equiv.jsonl") for n in ("a", "b"))
    with open(a) as f:
        rows = [json.loads(line) for line in f]
    assert "meta" in rows[0] and len(rows) == 13
    step = rows[1]
    for k in ("loss", "grad_norm", "depth", "ponder", "loads", "admitted"):
        assert step[k] is not None, k
    assert all(np.isfinite(r["loss"]) for r in rows[1:])

    p = run("perf_equiv.py", "--compare", a, b, "--tol", "0")
    assert p.returncode == 0 and "EQUAL" in p.stdout, p.stdout
    assert "max |diff| 0.000e+00" in p.stdout, p.stdout

    # and a real difference is caught
    with open(b) as f:
        lines = f.read().splitlines()
    r = json.loads(lines[3])
    r["loss"] += 1e-3
    r["loads"] += 1
    lines[3] = json.dumps(r)
    c = tmp_path / "c.jsonl"
    c.write_text("\n".join(lines) + "\n")
    p = run("perf_equiv.py", "--compare", a, str(c), "--tol", "1e-4")
    assert p.returncode == 1 and "DIFFERENT" in p.stdout, p.stdout
    assert "first at step 2: loads" in p.stdout, p.stdout


def test_profile_runs(tmp_path):
    cfg, corpus = tiny(tmp_path)
    out = tmp_path / "prof"
    p = run("perf_profile.py", "--config", cfg, "--data", corpus,
            "--steps", "4", "--warmup", "1", "--prof-warmup", "1",
            "--device", "cpu", "--out", str(out), "--table")
    assert p.returncode == 0, p.stdout + p.stderr
    trace = out / "trace.json.gz"
    assert trace.stat().st_size > 0
    text = (out / "profile.txt").read_text()
    for part in ("wall", "python", "syncs", "paging"):
        assert part in text, part

    p = run("perf_trace_summary.py", str(trace))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "steps           4 " in p.stdout, p.stdout
    for part in ("wall", "python / other", "syncs", "pool:experts fwd",
                 "pool:experts bwd", "paging", "top 10 kernels"):
        assert part in p.stdout, part


def test_bench_prints_one_line(tmp_path):
    cfg, corpus = tiny(tmp_path)
    p = run("perf_bench.py", "--config", cfg, "--data", corpus,
            "--steps", "4", "--warmup", "2", "--device", "cpu")
    assert p.returncode == 0, p.stdout + p.stderr
    lines = p.stdout.strip().splitlines()
    assert lines[-1].startswith("chars/s "), lines
    assert float(lines[-1].split()[1]) > 0
