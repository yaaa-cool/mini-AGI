"""tools/rig/rig.py, without a rig: argument parsing, the commands and remote
scripts --dry-run prints for every subcommand, `bash -n` over the remote
scripts, and lib.sh's config select run locally against a fake checkout."""

import importlib.util
import os
import re
import subprocess
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
    for name in ("smoke.sh", "train.sh", "eval-gen.sh"):
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
        assert cmds[0] == "gpu_free || exit 1", (name, cmds[0])


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
    assert f"bash {RUNS}/.rig/gpu-check.sh 85 1024" in cmds[1]
    assert "--query-compute-apps" in out and "systemctl is-active strata-swift" in out
    rc, out = dry(capsys, "gpu-check", "--max-temp", "80", "--max-used-mib", "2048")
    assert "gpu-check.sh 80 2048" in out


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
    assert (f"smoke.sh {RUNS}/w5-smoke small padded 15 "
            f"{RUNS}/w4-lanes/chat-train {RUNS}/w4-lanes/chat-val") in cmds[1]
    assert "RIG_REPO=~/train/mini-AGI" in cmds[1]
    assert "export MINAGI_DISPATCH=$DISPATCH" in out
    assert "tools/device_check.py --window 2048 --chunk 2048" in out
    assert "fresh_dir" in out and "GUARD_C=85" in out
    assert any(f"wait.sh {RUNS}/w5-smoke end 45" in c for c in cmds)
    assert cmds[-1].endswith(f"status.sh {RUNS}/w5-smoke'")
    rc, out = dry(capsys, "smoke", "--name", "s2", "--dispatch", "exact",
                  "--minutes", "5", "--no-wait")
    cmds = commands(out)
    assert f"smoke.sh {RUNS}/s2 small exact 5 " in cmds[-1]
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
    assert (f"train.sh {W} small padded {RUNS}/w4-A/weights data/train persona:1,chat:1 "
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
    assert f"train.sh {RUNS}/t - exact - - - train.py read data/replay-chat " in launch
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
    assert (f"eval-gen.sh {RUNS}/w5-C 3 800 {RUNS}/heldout.jsonl padded "
            f"A={RUNS}/w4-A/weights author={RUNS}/author/hf/weights") in cmd
    assert "tools/persona_generate.py" in out and "gpu_free" in out
    rc, out = dry(capsys, "eval", "gen", "--name", "e", "--model", "B=x",
                  "--shards", "2", "--max-chars", "400", "--dispatch", "exact")
    assert f"eval-gen.sh {RUNS}/e 2 400 {RUNS}/heldout.jsonl exact B={RUNS}/x" in out
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
