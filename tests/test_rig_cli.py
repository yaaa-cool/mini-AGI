"""tools/rig/rig.py, without a rig: argument parsing, the commands and remote
scripts --dry-run prints for every subcommand, `bash -n` over the remote
scripts, and lib.sh's config select run locally against a fake checkout."""

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "tools" / "rig" / "remote"
RUNS = "/mnt/data/runs/mini-agi"

_spec = importlib.util.spec_from_file_location("rig", ROOT / "tools" / "rig" / "rig.py")
rig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rig)


def fake_subprocess(run):
    return types.SimpleNamespace(run=run, PIPE=subprocess.PIPE)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A dry run must not run anything at all."""
    def refuse(*a, **k):
        raise AssertionError(f"subprocess.run called: {a}")
    monkeypatch.setattr(rig, "subprocess", fake_subprocess(refuse))


def dry(capsys, *argv):
    rc = rig.main(["--dry-run", *argv])
    return rc, capsys.readouterr().out


def commands(out):
    return [line[2:] for line in out.splitlines() if line.startswith("$ ")]


def test_remote_scripts_parse():
    scripts = sorted(REMOTE.glob("*.sh"))
    assert {p.name for p in scripts} >= {
        "lib.sh", "guarded.sh", "gpu-check.sh", "swift.sh", "config.sh", "test.sh",
        "smoke.sh", "train.sh", "wait.sh", "status.sh", "eval-gen.sh", "eval-status.sh"}
    for p in scripts:
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, f"{p.name}: {r.stderr}"


def test_gpu_free_check_heads_every_gpu_job():
    for name, check in (("smoke.sh", 'pick_gpus "$GPU" || exit 1'),
                        ("train.sh", 'pick_gpus "$GPU" || exit 1'),
                        ("eval-gen.sh", 'pick_gpus "$GPU" all || exit 1')):
        # the script's own top-level commands, past sourcing lib.sh, reading
        # the arguments and defining functions: the first is the check
        top, in_fn = [], False
        for line in (REMOTE / name).read_text().splitlines():
            if re.match(r"^\w+\(\) \{", line):
                in_fn = True
            elif in_fn:
                in_fn = line != "}"
            elif line and not line[0].isspace() and not line.startswith("#"):
                top.append(line)
        cmds = [c for c in top if not re.match(r"^(\. |shift|[A-Z_]+=)", c)]
        assert cmds[0] == check, (name, cmds[0])


def test_guarded_runner_keeps_w4_behaviour():
    text = (REMOTE / "guarded.sh").read_text()
    assert "GUARD_C=85" in text and 'kill -TERM $PID' in text
    assert "sleep 10" in text and '"$W/gpu.csv"' in text
    for f in ("pid", "start", "end", "guard.log", "read.log"):
        assert f'"$W/{f}"' in text


def test_gpu_check(capsys):
    rc, out = dry(capsys, "gpu-check")
    assert rc == 0
    cmds = commands(out)
    assert cmds[0].startswith("rsync -a ") and cmds[0].endswith(f"train@10.0.1.19:{RUNS}/.rig/")
    assert f"bash {RUNS}/.rig/gpu-check.sh all 85 1024" in cmds[1]
    assert "--query-compute-apps" in out and "systemctl is-active strata-swift" in out
    rc, out = dry(capsys, "gpu-check", "--max-temp", "80", "--max-used-mib", "2048")
    assert "gpu-check.sh all 80 2048" in out
    rc, out = dry(capsys, "gpu-check", "--gpu", "0")
    assert "gpu-check.sh 0 85 1024" in out


def test_swift(capsys):
    rc, out = dry(capsys, "swift", "stop")
    assert rc == 0
    (cmd,) = commands(out)
    assert "cc@10.0.1.19" in cmd and "bash -s -- stop" in cmd
    assert cmd.endswith("swift.sh")
    assert "sudo systemctl stop strata-swift" in out
    rc, out = dry(capsys, "swift", "start", "--wait")
    assert "poll GET http://10.0.1.19:8080/v1/models" in out
    with pytest.raises(SystemExit):
        rig.main(["--dry-run", "swift", "restart"])


def test_sync(capsys):
    rc, out = dry(capsys, "sync")
    assert rc == 0
    cmds = commands(out)
    code = cmds[0]
    for x in (".venv", "data", "data_*", "runs", ".git", ".claude", "__pycache__",
              ".pytest_cache", ".codegraph", "/weights/", "/weights-small/", "/config.yaml"):
        assert f"--exclude={x}" in code, x
    assert code.endswith(" train@10.0.1.19:train/mini-AGI/")
    # the Mac's config.yaml goes to the reference copy, then the active one is re-applied
    assert any(c.endswith(f"{RUNS}/.rig/config.reference.yaml") for c in cmds)
    assert cmds[-1].endswith("/.rig/config.sh'")
    assert "select_config" in out and "REFUSED" in out
    assert "test.sh" not in out
    rc, out = dry(capsys, "sync", "--config", "small", "--test")
    cmds = commands(out)
    assert any("config.sh small" in c for c in cmds)
    assert any(c.endswith("/.rig/test.sh'") for c in cmds)
    assert 'CUDA_VISIBLE_DEVICES=""' in out


def test_smoke(capsys):
    rc, out = dry(capsys, "smoke")
    assert rc == 0
    cmds = commands(out)
    assert (f"smoke.sh {RUNS}/w5-smoke small padded auto 15 "
            f"{RUNS}/w4-lanes/chat-train {RUNS}/w4-lanes/chat-val") in cmds[1]
    assert 'RIG_REPO="$HOME"/train/mini-AGI' in cmds[1]
    assert "export MINAGI_DISPATCH=$DISPATCH" in out
    assert "tools/device_check.py --window 2048 --chunk 2048" in out
    assert "fresh_dir" in out and "GUARD_C=85" in out
    assert any(f"wait.sh {RUNS}/w5-smoke end 45" in c for c in cmds)
    assert cmds[-1].endswith(f"status.sh {RUNS}/w5-smoke'")
    rc, out = dry(capsys, "smoke", "--name", "s2", "--dispatch", "exact",
                  "--minutes", "5", "--no-wait")
    cmds = commands(out)
    assert f"smoke.sh {RUNS}/s2 small exact auto 5 " in cmds[-1]
    assert "wait.sh" not in " ".join(cmds)


TRAIN = ["train", "--name", "w5-persona", "--data", "data/train",
         "--mix", "persona=0.5,chat=0.5", "--held-out", "data/val",
         "--config", "small", "--from-weights", "w4-A/weights",
         "--lr", "3e-5", "--lr-reset", "--minutes", "180"]


def test_train(capsys):
    rc, out = dry(capsys, *TRAIN)
    assert rc == 0
    cmds = commands(out)
    W = f"{RUNS}/w5-persona"
    launch = next(c for c in cmds if "train.sh" in c)
    assert (f"train.sh {W} small padded auto {RUNS}/w4-A/weights data/train persona:1,chat:1 "
            f"train.py read {W}/mix --save --weights-dir {W}/weights --held-out data/val "
            f"--sample-every 10 --sample-log {W}/samples.txt --minutes 180 --lr 3e-5 --lr-reset") in launch
    assert "--drop-quietest" not in out
    # fresh dir, a cp -a of the weights, the exact command and config recorded
    assert 'fresh_dir "$W"' in out and 'cp -a "$FROM" "$W/weights"' in out
    assert '> "$w/cmd"' in out and 'cp "$REPO/config.yaml" "$w/config.yaml"' in out
    assert "build_mix" in out and "nohup bash" in out
    # --lr-reset waits for the first rate line
    assert cmds[-1].endswith(f"wait.sh {W} rate 30'")


def test_train_options(capsys):
    rc, out = dry(capsys, "train", "--name", "t", "--data", "data/replay-chat",
                  "--held-out", "data/val", "--max-chars", "30e6", "--dispatch", "exact",
                  "--no-newborn-lr", "--newborn-half", "8000", "--sample-every", "5")
    assert rc == 0
    launch = next(c for c in commands(out) if "train.sh" in c)
    assert f"train.sh {RUNS}/t - exact auto - - - train.py read data/replay-chat " in launch
    assert "--max-chars 30e6" in launch and "--no-newborn-lr" in launch
    assert "--newborn-half 8000" in launch and "--sample-every 5" in launch
    assert "--minutes" not in launch and "--lr-reset" not in launch
    assert not any("wait.sh" in c for c in commands(out))
    rc, out = dry(capsys, "train", "--name", "t", "--data", "d", "--held-out", "v",
                  "--minutes", "60", "--newborn-lr")
    assert "--newborn-lr" in next(c for c in commands(out) if "train.sh" in c)


def test_train_refusals(capsys):
    rc, out = dry(capsys, "train", "--name", "t", "--data", "d", "--held-out", "v")
    assert rc == 2 and "--minutes or --max-chars" in out
    for bad in (["--drop-quietest", "3"], ["--newborn-lr", "--no-newborn-lr"],
                ["--mix", "persona=0.3,chat=0.7"], ["--lr", "fast"]):
        with pytest.raises(SystemExit):
            rig.main(["--dry-run", "train", "--name", "t", "--data", "d",
                      "--held-out", "v", "--minutes", "5", *bad])
    with pytest.raises(SystemExit):
        rig.main(["--dry-run", "train", "--name", "../x", "--data", "d",
                  "--held-out", "v", "--minutes", "5"])


def test_mix_spec():
    assert rig.mix_spec("persona=0.5,chat=0.5") == "persona:1,chat:1"
    assert rig.mix_spec("persona=0.25,chat=0.75") == "persona:1,chat:3"
    assert rig.mix_spec("persona=2,chat=1,code=1") == "persona:2,chat:1,code:1"
    for bad in ("persona", "persona=0.3,chat=0.7", "a=1,b=20", "a=0,b=1"):
        with pytest.raises(Exception):
            rig.mix_spec(bad)


def test_status(capsys):
    rc, out = dry(capsys, "status", "w5-persona")
    assert rc == 0
    assert commands(out)[-1].endswith(f"status.sh {RUNS}/w5-persona'")
    for s in ("maxT=", "swThermalActive", "guard hits", "char/s",
              "# held-out at the start, spread over the whole set:",
              "^held-out loss", "ETA", "state: "):
        assert s in out, s
    rc, out = dry(capsys, "status", "w5-persona", "--watch", "60")
    assert "repeats every 60 s" in out


def test_eval_gen(capsys):
    rc, out = dry(capsys, "eval", "gen", "--name", "w5-C", "--model", "A=w4-A/weights",
                  "--model", "author=/mnt/data/runs/mini-agi/author/hf/weights")
    assert rc == 0
    cmd = commands(out)[-1]
    assert (f"eval-gen.sh {RUNS}/w5-C 3 800 {RUNS}/heldout.jsonl padded auto "
            f"A={RUNS}/w4-A/weights author={RUNS}/author/hf/weights") in cmd
    assert "tools/persona_generate.py" in out and "gpu_free" in out
    rc, out = dry(capsys, "eval", "gen", "--name", "e", "--model", "B=x",
                  "--shards", "2", "--max-chars", "400", "--dispatch", "exact")
    assert f"eval-gen.sh {RUNS}/e 2 400 {RUNS}/heldout.jsonl exact auto B={RUNS}/x" in out
    with pytest.raises(SystemExit):
        rig.main(["--dry-run", "eval", "gen", "--name", "e", "--model", "noweights"])


def test_eval_status_and_score(capsys):
    rc, out = dry(capsys, "eval", "status", "--name", "w5-C")
    assert rc == 0 and commands(out)[-1].endswith(f"eval-status.sh {RUNS}/w5-C'")
    rc, out = dry(capsys, "eval", "score", "--name", "w5-C", "--no-judge")
    assert rc == 0
    cmds = commands(out)
    assert cmds[1] == f"scp -q 'train@10.0.1.19:{RUNS}/w5-C/replies.*.jsonl' {ROOT}/runs/w5-C/"
    score = cmds[-1]
    assert "tools/persona_score.py" in score and "--judge-url http://10.0.1.19:8080" in score
    assert score.endswith("--no-judge") and f"--out-dir {ROOT}/runs/w5-C/score" in score


def test_fetch(capsys):
    rc, out = dry(capsys, "fetch", "w5-persona", "--dest", "/tmp/x")
    assert rc == 0
    cmds = commands(out)
    for f in ("read.log", "samples.txt", "expert_rates.jsonl", "gpu.csv", "cmd",
              "config.yaml", "guard.log", "end"):
        assert f"scp -q train@10.0.1.19:{RUNS}/w5-persona/{f} /tmp/x/{f}" in cmds


def test_fetch_skips_missing(capsys, monkeypatch, tmp_path):
    def fake(cmd, **k):
        name = cmd[-1].rsplit("/", 1)[-1]
        if name in ("guard.log", "end"):
            return subprocess.CompletedProcess(cmd, 1)
        Path(cmd[-1]).write_text("x")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(rig, "subprocess", fake_subprocess(fake))
    assert rig.main(["fetch", "w5-persona", "--dest", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "skip guard.log" in out and "skip end" in out
    assert (tmp_path / "read.log").exists()


def test_global_flags(capsys, monkeypatch):
    monkeypatch.setenv("MINAGI_RIG_HOST", "me@rig")
    rc, out = dry(capsys, "--runs", "/r", "--repo", "/srv/m", "gpu-check")
    cmds = commands(out)
    assert cmds[0].endswith("me@rig:/r/.rig/")
    assert "RIG_REPO=/srv/m RIG_RUNS=/r" in cmds[1]


def test_home_paths_reach_the_rig_expanded(capsys, tmp_path):
    """A ~/ path goes as "$HOME"/..., so the rig's shell expands it (a quoted
    ~ arrives literal and train.sh refuses it)."""
    assert rig.rquote("~/w4 A/weights") == "\"$HOME\"/'w4 A/weights'"
    assert rig.rquote("A=~/w") == 'A="$HOME"/w'
    assert rig.rquote("/abs/~/x") == "'/abs/~/x'"
    assert rig.rquote("a'b") == shlex.quote("a'b")
    words = ["~/a b", "A=~/w", "~/", "x~/y", "$(touch pwned)"]
    r = subprocess.run(["bash", "-c", "printf '%s\\n' " + " ".join(map(rig.rquote, words))],
                       env=dict(os.environ, HOME=str(tmp_path)), cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.stdout.splitlines() == [f"{tmp_path}/a b", f"A={tmp_path}/w", f"{tmp_path}/",
                                     "x~/y", "$(touch pwned)"]
    assert not (tmp_path / "pwned").exists()
    rc, out = dry(capsys, "train", "--name", "t", "--data", "~/d", "--held-out", "~/v",
                  "--from-weights", "~/w", "--minutes", "5")
    launch = next(c for c in commands(out) if "train.sh" in c)
    assert 'padded auto "$HOME"/w - - train.py read "$HOME"/d' in launch
    assert '--held-out "$HOME"/v' in launch
    rc, out = dry(capsys, "eval", "gen", "--name", "e", "--model", "A=~/w")
    assert 'A="$HOME"/w' in commands(out)[-1]


# ---- lib.sh's config select, run for real against a fake checkout

def config_sh(tmp_path, *args):
    env = dict(os.environ, RIG_REPO=str(tmp_path / "repo"), RIG_RUNS=str(tmp_path / "runs"))
    r = subprocess.run(["bash", str(REMOTE / "config.sh"), *args], env=env,
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


@pytest.fixture
def fake_rig(tmp_path):
    repo, rigdir = tmp_path / "repo", tmp_path / "runs" / ".rig"
    repo.mkdir()
    rigdir.mkdir(parents=True)
    (repo / "config.yaml").write_text("d_model: 512\n")          # what a sync leaves
    (repo / "config.small.yaml").write_text("d_model: 384\n")
    (rigdir / "config.reference.yaml").write_text("d_model: 512\n")
    return tmp_path


def test_config_select_and_reapply(fake_rig):
    rc, out = config_sh(fake_rig, "small")
    assert rc == 0, out
    assert (fake_rig / "repo" / "config.yaml").read_text() == "d_model: 384\n"
    assert (fake_rig / "runs" / ".rig" / "active-config").read_text() == "small\n"
    # a sync that brought config.yaml back to reference: re-apply restores small
    (fake_rig / "repo" / "config.yaml").write_text("d_model: 512\n")
    rc, out = config_sh(fake_rig)
    assert rc == 0 and "config: small" in out
    assert (fake_rig / "repo" / "config.yaml").read_text() == "d_model: 384\n"


def test_config_refused_while_a_run_is_live(fake_rig):
    run = fake_rig / "runs" / "w5-a"
    run.mkdir()
    live = subprocess.Popen(["sleep", "30"])
    try:
        (run / "pid").write_text(f"{live.pid}\n")
        rc, out = config_sh(fake_rig, "small")
        assert rc == 3 and "REFUSED" in out and str(run) in out
        assert (fake_rig / "repo" / "config.yaml").read_text() == "d_model: 512\n"
        # the same config is no change, so it is fine while the run lives
        rc, out = config_sh(fake_rig, "reference")
        assert rc == 0, out
        (run / "end").write_text("exit 0\n")
        rc, out = config_sh(fake_rig, "small")
        assert rc == 0, out
    finally:
        live.kill()
        live.wait()


def test_config_reference_needs_a_sync(fake_rig):
    (fake_rig / "runs" / ".rig" / "config.reference.yaml").unlink()
    rc, out = config_sh(fake_rig, "reference")
    assert rc == 2 and "rig.py sync" in out


SMALL_NOW = "# d_model 512 -> 384, the cut is in width\nmodel:\n  d_model: 384    # SMALL\n  layers: 6\n"
SMALL_OLD = "# d_model 512 -> 384\nmodel:\n  d_model: 384    # SMALL\n  layers: 4\n"


def test_config_detected_by_d_model_not_bytes(fake_rig):
    """No active-config yet: an older small copy (config.small.yaml changed
    upstream since) is still small, and is not overwritten with reference."""
    repo = fake_rig / "repo"
    (repo / "config.small.yaml").write_text(SMALL_NOW)
    (repo / "config.yaml").write_text(SMALL_OLD)
    rc, out = config_sh(fake_rig)
    assert rc == 0 and "config: small (d_model: 384" in out, out
    assert (repo / "config.yaml").read_text() == SMALL_NOW
    assert (fake_rig / "runs" / ".rig" / "active-config").read_text() == "small\n"


def test_config_detection_skips_comments(fake_rig):
    repo = fake_rig / "repo"
    (repo / "config.yaml").write_text("# d_model 384 in the small one\nmodel:\n  d_model: 512\n")
    rc, out = config_sh(fake_rig, "-")
    assert rc == 0 and "kept (d_model: 512)" in out, out
    rc, out = config_sh(fake_rig)
    assert rc == 0 and "config: reference (d_model: 512)" in out, out


def test_config_unknown_width_asks_for_config(fake_rig):
    repo = fake_rig / "repo"
    (repo / "config.yaml").write_text("model:\n  d_model: 256\n")
    rc, out = config_sh(fake_rig)
    assert rc == 2 and "d_model '256'" in out and "--config small|reference" in out, out
    assert (repo / "config.yaml").read_text() == "model:\n  d_model: 256\n"
    assert not (fake_rig / "runs" / ".rig" / "active-config").exists()
    rc, out = config_sh(fake_rig, "small")
    assert rc == 0 and (repo / "config.yaml").read_text() == "d_model: 384\n"


# ---- status.sh's ETA, run for real against a fake run dir (needs GNU date)

def gnu_date_path(tmp_path):
    if subprocess.run(["date", "-d", "@0"], capture_output=True).returncode == 0:
        return os.environ["PATH"]
    gdate = shutil.which("gdate")
    if not gdate:
        pytest.skip("no GNU date")
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "date").symlink_to(gdate)
    return f"{shim}:{os.environ['PATH']}"


def test_status_eta_for_max_chars(tmp_path):
    env = dict(os.environ, PATH=gnu_date_path(tmp_path), TZ="UTC")
    w = tmp_path / "t"
    w.mkdir()
    (w / "start").write_text("2026-10-17T10:00:00+00:00\n")
    (w / "cmd").write_text("# x\nMINAGI_DISPATCH=padded PYTHONUNBUFFERED=1 py train.py read d "
                           "--max-chars 30e6 --sample-every 10\n")
    (w / "read.log").write_text("  10.0M characters read in earlier sessions\n")
    r = subprocess.run(["bash", str(REMOTE / "status.sh"), str(w)], env=env,
                       capture_output=True, text=True)
    assert "no estimate until samples.txt has a check" in r.stdout, r.stdout + r.stderr
    (w / "samples.txt").write_text(
        "step 500   15.0M of 500M characters (3.00%)   30 min   12 experts\n"
        "context 4,096 characters of 32,768   reading 9,000 char/s\n"
        "step 1,000   20.0M of 500M characters (4.00%)   60 min   12 experts\n"
        "context 4,096 characters of 32,768   reading 5,000 char/s   writing 40.0 char/s\n")
    r = subprocess.run(["bash", str(REMOTE / "status.sh"), str(w)], env=env,
                       capture_output=True, text=True)
    # 10M of 30M read this session at the 60 min check; 20M more at 5,000 char/s
    assert ("ETA ~2026-10-17T12:06:40+00:00 (an estimate: --max-chars 30e6, "
            "10.0M read at the last check, at its 5000 char/s") in r.stdout, r.stdout + r.stderr
    assert r.stdout.splitlines()[-1].startswith("state: dead")


# ---- the two-card rig: a fake nvidia-smi first on PATH, the scripts run for real

FAKE_NVIDIA_SMI = r'''
import json, os, sys
state = json.load(open(os.environ["FAKE_GPUS"]))
units = {"clocks.sm": " MHz", "memory.used": " MiB", "memory.total": " MiB",
         "power.limit": " W", "power.draw": " W", "utilization.gpu": " %",
         "used_memory": " MiB"}
card, rows, fields, fmt = None, None, [], ""
args = iter(sys.argv[1:])
for a in args:
    if a == "-i":
        card = next(args)
    elif a.startswith("--query-gpu="):
        fields, rows = a.split("=", 1)[1].split(","), state["gpus"]
    elif a.startswith("--query-compute-apps="):
        fields, rows = a.split("=", 1)[1].split(","), state["apps"]
    elif a.startswith("--format="):
        fmt = a
if card is not None:
    rows = [g for g in rows if str(g["index"]) == card]
    if not rows:
        print("No devices were found")
        sys.exit(6)
if "noheader" not in fmt:
    print(", ".join(fields))
for r in rows:
    print(", ".join(f"{r[f]}{'' if 'nounits' in fmt else units.get(f, '')}" for f in fields))
'''


def set_cards(env, temps=(56, 50), used=(9, 1), apps=()):
    """The fake cards' state; apps are (card, pid, name, MiB). Replaced whole,
    so a script reading it mid-run never sees half a file."""
    gpus = [{"index": i, "uuid": f"GPU-{i}a1b2", "name": "NVIDIA GeForce RTX 3060",
             "timestamp": "2026/10/19 10:00:00.000", "temperature.gpu": t, "clocks.sm": 405,
             "memory.used": u, "memory.total": 12288, "power.limit": "100.00",
             "power.draw": "20.00", "utilization.gpu": 0,
             "clocks_event_reasons.sw_thermal_slowdown": "Not Active",
             "clocks_event_reasons.hw_thermal_slowdown": "Not Active"}
            for i, (t, u) in enumerate(zip(temps, used))]
    state = {"gpus": gpus, "apps": [{"gpu_uuid": f"GPU-{c}a1b2", "pid": pid, "process_name": name,
                                     "used_memory": mib} for c, pid, name, mib in apps]}
    path = Path(env["FAKE_GPUS"])
    path.with_suffix(".tmp").write_text(json.dumps(state))
    path.with_suffix(".tmp").replace(path)


@pytest.fixture
def two_cards(tmp_path):
    """Two RTX 3060s behind a fake nvidia-smi, a fake rig checkout and run root,
    a `sleep` that takes 0.2 s (the guard's 10 s and the launch's 2 s), and
    GNU date where the Mac has it as gdate."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    smi = bin_ / "nvidia-smi"
    smi.write_text(f"#!{sys.executable}\n{FAKE_NVIDIA_SMI}")
    (bin_ / "sleep").write_text(f"#!/bin/sh\nexec {shutil.which('sleep')} 0.2\n")
    for f in (smi, bin_ / "sleep"):
        f.chmod(0o755)
    if subprocess.run(["date", "-d", "@0"], capture_output=True).returncode and shutil.which("gdate"):
        (bin_ / "date").symlink_to(shutil.which("gdate"))
    repo, rigdir = tmp_path / "repo", tmp_path / "runs" / ".rig"
    repo.mkdir()
    rigdir.mkdir(parents=True)
    (repo / "config.yaml").write_text("model:\n  d_model: 512   # width of the trunk\n")
    shutil.copy(REMOTE / "guarded.sh", rigdir / "guarded.sh")
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", FAKE_GPUS=str(tmp_path / "gpus.json"),
               RIG_REPO=str(repo), RIG_RUNS=str(tmp_path / "runs"), TZ="UTC")
    set_cards(env)
    return env


def need_gnu_date(env):
    if subprocess.run(["date", "-Is"], env=env, capture_output=True).returncode:
        pytest.skip("no GNU date")


def lib(env, code):
    r = subprocess.run(["bash", "-c", f". {shlex.quote(str(REMOTE / 'lib.sh'))}; {code}"],
                       env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def bash(env, script, *args, timeout=30):
    r = subprocess.run(["bash", str(REMOTE / script), *map(str, args)], env=env,
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def wait_for(path, timeout=15):
    end = time.time() + timeout
    while not path.exists() and time.time() < end:
        time.sleep(0.1)
    assert path.exists(), path


def live_run(env, name, card):
    """A guarded run dir whose process is alive, launched on card."""
    d = Path(env["RIG_RUNS"]) / name
    d.mkdir()
    p = subprocess.Popen([shutil.which("sleep"), "60"])
    (d / "pid").write_text(f"{p.pid}\n")
    (d / "gpu").write_text(f"{card}\n")
    return d, p


def test_fake_nvidia_smi_answers_like_the_rig(two_cards):
    """The rig's own answer (2026-10-19), so the fake is the shape the scripts parse."""
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,memory.used,"
                        "memory.total,power.limit,clocks_event_reasons.sw_thermal_slowdown",
                        "--format=csv,noheader"], env=two_cards, capture_output=True, text=True)
    assert r.stdout == ("NVIDIA GeForce RTX 3060, 56, 405 MHz, 9 MiB, 12288 MiB, 100.00 W, Not Active\n"
                        "NVIDIA GeForce RTX 3060, 50, 405 MHz, 1 MiB, 12288 MiB, 100.00 W, Not Active\n")


def test_gpu_free_per_card(two_cards):
    env = two_cards
    assert lib(env, "gpu_free 0")[0] == 0 and lib(env, "gpu_free 1")[0] == 0
    # strata-swift-like load on card 1: card 0 stays free
    set_cards(env, used=(9, 11600), apps=[(1, 4242, "llama-server", 11590)])
    rc, out = lib(env, "gpu_free 0")
    assert rc == 0 and "GPU 0 free: 9 MiB used, 56 C" in out, out
    rc, out = lib(env, "gpu_free 1")
    assert rc == 1 and "GPU 1 BUSY: 11600 MiB" in out and "4242, llama-server, 11590" in out, out
    # the temperature is the card's own
    set_cards(env, temps=(85, 50))
    rc, out = lib(env, "gpu_free 0")
    assert rc == 1 and "GPU 0 HOT: 85 C >= 85 C" in out, out
    assert lib(env, "gpu_free 1")[0] == 0
    rc, out = lib(env, "gpu_free 2")
    assert rc == 1 and "GPU 2 CHECK FAILED" in out, out
    # a live guarded run holds its card before its model shows up as an app
    set_cards(env)
    d, p = live_run(env, "w5-a", 0)
    try:
        rc, out = lib(env, "gpu_free 0")
        assert rc == 1 and f"GPU 0 BUSY: live guarded runs on it:\n  {d}" in out, out
        assert lib(env, "gpu_free 1")[0] == 0
    finally:
        p.kill()
        p.wait()


def test_auto_picks_a_free_card(two_cards):
    env = two_cards
    show = 'pick_gpus {} {}; rc=$?; echo "GPUS=${{GPUS[*]}}"; exit $rc'
    rc, out = lib(env, show.format("auto", ""))
    assert rc == 0 and "GPUS=0\n" in out, out
    rc, out = lib(env, show.format("auto", "all"))
    assert rc == 0 and "GPUS=0 1\n" in out, out
    set_cards(env, apps=[(0, 4242, "python", 3000)])
    rc, out = lib(env, show.format("auto", ""))
    assert rc == 0 and "GPU 0 BUSY" in out and "GPUS=1\n" in out, out
    rc, out = lib(env, show.format("auto", "all"))
    assert rc == 0 and "GPUS=1\n" in out, out
    assert lib(env, show.format("0", ""))[0] == 1
    rc, out = lib(env, show.format("0,1", "all"))
    assert rc == 1 and "GPU 0 BUSY" in out, out
    assert lib(env, show.format("1", ""))[0] == 0
    set_cards(env, temps=(56, 88), apps=[(0, 4242, "python", 3000)])
    rc, out = lib(env, show.format("auto", "all"))
    assert rc == 1 and "NO FREE GPU" in out, out


def test_guard_fires_on_its_own_card(two_cards):
    env = two_cards
    need_gnu_date(env)
    w = Path(env["RIG_RUNS"]) / "hot"
    p = subprocess.Popen(["bash", str(REMOTE / "guarded.sh"), str(w), "1",
                          shutil.which("sleep"), "60"], env=env)
    try:
        wait_for(w / "gpu.csv")
        time.sleep(1)
        assert not (w / "end").exists()
        set_cards(env, temps=(56, 85))
        p.wait(timeout=15)
    finally:
        p.kill()
    assert (w / "end").read_text().startswith("exit 143")
    assert "TEMP GUARD: card 1 85C >= 85" in (w / "guard.log").read_text()
    rows = (w / "gpu.csv").read_text().splitlines()
    assert rows[0].endswith(",hw_thermal,gpu")
    assert all(r.split(", ")[1] in ("50", "85") and r.endswith(", 1") for r in rows[1:]), rows


def test_guard_ignores_the_other_card(two_cards):
    """item-1 decision: only the run's own card trips its guard."""
    env = two_cards
    need_gnu_date(env)
    set_cards(env, temps=(60, 95))
    w = Path(env["RIG_RUNS"]) / "cool"
    rc, out = bash(env, "guarded.sh", w, 0, shutil.which("sleep"), 1)
    assert rc == 0, out
    assert (w / "end").read_text().startswith("exit 0")
    assert not (w / "guard.log").exists()
    rows = (w / "gpu.csv").read_text().splitlines()[1:]
    assert rows and all(r.split(", ")[1] == "60" and r.endswith(", 0") for r in rows), rows


def fake_python(env, tmp_path):
    py = tmp_path / "py"
    py.write_text('#!/bin/sh\necho "card=$CUDA_VISIBLE_DEVICES order=$CUDA_DEVICE_ORDER '
                  'dispatch=$MINAGI_DISPATCH args $*"\n')
    py.chmod(0o755)
    return dict(env, RIG_PY=str(py))


def test_train_launch_pins_the_auto_card(two_cards, tmp_path):
    env = fake_python(two_cards, tmp_path)
    need_gnu_date(env)
    set_cards(env, used=(3009, 1), apps=[(0, 4242, "python", 3000)])
    w = Path(env["RIG_RUNS"]) / "t"
    rc, out = bash(env, "train.sh", w, "-", "padded", "auto", "-", "-", "-", "train.py", "read", "d")
    assert rc == 0, out
    assert "GPU 0 BUSY" in out and "auto: card 1" in out and f"launched {w} on card 1" in out, out
    wait_for(w / "end")
    assert (w / "gpu").read_text() == "1\n"
    cmd = (w / "cmd").read_text()
    assert "card 1\n" in cmd and "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MINAGI_DISPATCH=padded" in cmd
    assert "card=1 order=PCI_BUS_ID dispatch=padded args train.py read d" in (w / "read.log").read_text()
    # a named busy card is refused, and nothing is launched
    rc, out = bash(env, "train.sh", w.with_name("t2"), "-", "padded", "0", "-", "-", "-", "train.py")
    assert rc == 1 and "GPU 0 BUSY" in out and not w.with_name("t2").exists(), out


def test_eval_gen_spreads_shards_over_the_cards(two_cards, tmp_path):
    env = fake_python(two_cards, tmp_path)
    need_gnu_date(env)
    bin_ = Path(env["PATH"].split(":")[0])
    if subprocess.run(["split", "--version"], capture_output=True).returncode:
        if not shutil.which("gsplit"):
            pytest.skip("no GNU split")
        (bin_ / "split").symlink_to(shutil.which("gsplit"))
    runs = Path(env["RIG_RUNS"])
    (runs / "heldout.jsonl").write_text("".join(f'{{"id": {i}}}\n' for i in range(6)))
    for m in ("A", "B"):
        (runs / m).mkdir()
    e = runs / "e"
    rc, out = bash(env, "eval-gen.sh", e, 3, 800, runs / "heldout.jsonl", "padded", "auto",
                   f"A={runs / 'A'}", f"B={runs / 'B'}")
    assert rc == 0 and "auto: card 0 1" in out, out
    cards = {}
    for d in sorted(e.glob("gen-*")):
        wait_for(d / "end")
        cards[d.name] = d.joinpath("gpu").read_text().strip()
        assert f"card={cards[d.name]} " in (d / "read.log").read_text()
    assert cards == {"gen-A-0": "0", "gen-A-1": "1", "gen-A-2": "0",
                     "gen-B-0": "1", "gen-B-1": "0", "gen-B-2": "1"}


def test_gpu_check_two_cards(two_cards):
    env = two_cards
    rc, out = bash(env, "gpu-check.sh", "all", 85, 1024)
    assert rc == 0, out
    assert "config: reference (d_model: 512)\n" in out, out
    for c in ("0", "1"):
        assert f"== card {c}\n" in out and f"GPU {c} free" in out
    assert "free cards: 0 1" in out
    assert "0, NVIDIA GeForce RTX 3060, 56, 405 MHz, 9 MiB, 12288 MiB, 100.00 W, Not Active" in out
    set_cards(env, used=(3009, 1), apps=[(0, 4242, "python", 3000)])
    d, p = live_run(env, "w5-a", 0)
    try:
        rc, out = bash(env, "gpu-check.sh", "all", 85, 1024)
        assert rc == 0 and "free cards: 1\n" in out, out
        assert "compute apps:\n  4242, python, 3000 MiB\nGPU 0 BUSY" in out, out
        assert f"  {d} (card 0)" in out
        rc, out = bash(env, "gpu-check.sh", 0, 85, 1024)
        assert rc == 1 and "== card 1" not in out and "free cards: none" in out, out
        assert bash(env, "gpu-check.sh", 1, 85, 1024)[0] == 0
        set_cards(env, temps=(56, 90), used=(3009, 1), apps=[(0, 4242, "python", 3000)])
        rc, out = bash(env, "gpu-check.sh", "all", 85, 1024)
        assert rc == 1 and "GPU 1 HOT: 90 C" in out and "free cards: none" in out, out
    finally:
        p.kill()
        p.wait()


def test_status_shows_the_card_and_both_temps(two_cards):
    w = Path(two_cards["RIG_RUNS"]) / "t"
    w.mkdir()
    (w / "gpu").write_text("1\n")
    rc, out = bash(two_cards, "status.sh", w)
    assert "card: 1\n" in out and "now: card 0 56 C, card 1 50 C\n" in out, out


def test_side_by_side_runs_share_the_active_config(two_cards):
    env = two_cards
    rigdir = Path(env["RIG_RUNS"]) / ".rig"
    repo = Path(env["RIG_REPO"])
    (repo / "config.small.yaml").write_text("d_model: 384\n")
    shutil.copy(repo / "config.yaml", rigdir / "config.reference.yaml")
    (rigdir / "active-config").write_text("reference\n")
    a, pa = live_run(env, "w5-a", 0)
    b, pb = live_run(env, "w5-b", 1)
    try:
        rc, out = bash(env, "config.sh", "small")
        assert rc == 3 and f"  {a}\n" in out and f"  {b}\n" in out, out
        assert "a second run side by side on the other" in out
        assert "card must use the active config, reference)" in out
        assert bash(env, "config.sh", "reference")[0] == 0
    finally:
        for p in (pa, pb):
            p.kill()
            p.wait()


def test_gpu_flags_dry_run(capsys):
    rc, out = dry(capsys, "smoke", "--gpu", "1", "--no-wait")
    assert rc == 0 and f"smoke.sh {RUNS}/w5-smoke small padded 1 15 " in commands(out)[-1]
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$card" in out
    assert 'echo "$card" > "$w/gpu"' in out and 'nvidia-smi -i "$CARD"' in out
    rc, out = dry(capsys, "train", "--name", "t", "--data", "d", "--held-out", "v",
                  "--minutes", "5", "--gpu", "0")
    assert f"train.sh {RUNS}/t - padded 0 - - - train.py" in commands(out)[-1]
    assert "CUDA_VISIBLE_DEVICES=$card" in out
    rc, out = dry(capsys, "eval", "gen", "--name", "e", "--model", "A=x", "--gpu", "0,1")
    assert f"heldout.jsonl padded 0,1 A={RUNS}/x" in commands(out)[-1]
    assert "CUDA_VISIBLE_DEVICES=$card" in out and 'pick_gpus "$GPU" all' in out
    for bad in (["smoke", "--gpu", "0,1"], ["smoke", "--gpu", "x"],
                ["eval", "gen", "--name", "e", "--model", "A=x", "--gpu", "0,0"],
                ["eval", "gen", "--name", "e", "--model", "A=x", "--gpu", "0,"],
                ["gpu-check", "--gpu", "auto"]):
        with pytest.raises(SystemExit):
            rig.main(["--dry-run", *bad])
