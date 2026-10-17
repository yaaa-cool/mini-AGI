#!/usr/bin/env python3
"""
rig.py - drive training on the Dell rig (RTX 3060) from the Mac.

    python3 tools/rig/rig.py [--host H] [--dry-run] <subcommand> ...

    gpu-check [--max-temp 85] [--max-used-mib 1024]
    swift stop|start|status [--wait]
    sync [--config small|reference] [--test]
    smoke [--name w5-smoke] [--config small] [--dispatch padded|exact] [--minutes 15] [--no-wait]
    train --name N --data DIR --held-out DIR [--mix persona=0.5,chat=0.5]
          [--config small|reference] [--from-weights DIR] [--lr X] [--lr-reset]
          [--minutes M] [--max-chars N] [--sample-every K] [--dispatch padded|exact]
          [--newborn-lr | --no-newborn-lr] [--newborn-half N]
    status N [--watch SEC]
    eval gen --name E --model NAME=WEIGHTS [--model ...] [--shards 3] [--max-chars 800]
    eval status --name E
    eval score --name E [--no-judge]
    fetch N [--dest runs/N]

Every piece of rig-side logic is a script in tools/rig/remote/, uploaded to
$RUNS/.rig/ and run as `ssh <host> bash $RUNS/.rig/<script> <args>`; nothing
here is more than a command line. --dry-run prints every rsync/scp/ssh
command and the text of each remote script it would run, and runs nothing.
Stdlib only, so the system python3 runs it.

Run dirs are $RUNS/<name> and hold what guarded.sh writes (read.log, gpu.csv,
pid, start, end, guard.log) plus samples.txt, expert_rates.jsonl, weights/,
cmd (the exact command) and config.yaml (the config it ran with).

Decisions on the points the spec left open:
- Dispatch. MINAGI_DISPATCH=padded for every GPU job (smoke, train, eval
  gen); --dispatch exact opts out and sets MINAGI_DISPATCH=exact explicitly,
  so `cmd` always says which.
- Fresh dirs. smoke and train refuse an existing $RUNS/<name> (a reused
  --weights-dir resumes); --from-weights is `cp -a`'d into the new dir. There
  is no resume through this tool.
- Relative paths. --from-weights and eval --model weights are relative to
  --runs; --data and --held-out are relative to --repo (train.py's cwd).
- Bounded runs. train needs --minutes or --max-chars: nothing unattended reads
  the whole corpus by accident. --drop-quietest is not exposed (it waits for
  N on stdin).
- --mix LANE=W,... becomes a folder per share under $RUNS/<name>/mix, because
  `read` gives every top-level folder an equal share: weights must reduce to
  small whole ratios (0.5,0.5 -> 1:1; 0.25,0.75 -> 1:3; at most 12 folders).
  The FIRST lane is linked whole; every other folder gets files up to the
  size of one of the first lane's folders, as replay.sh did. Hard links, or
  symbolic links if the corpus is on another disk (read follows both).
- sync. The rsync excludes the cookbook's list (.venv data data_* runs .git
  .claude __pycache__ .pytest_cache .codegraph) plus weights/ and
  weights-small/: the upstream weights snapshot is not in git (they are
  gitignored and published separately), so the code needs neither, and a
  local weights dir must never land on the rig's checkpoints. .minagi-dry-*
  (dry-read scratch) is excluded for the same reason.
- The config.yaml gotcha. sync excludes config.yaml from the code rsync, so
  the rig's copy is never reset under a live run; the Mac's config.yaml goes
  to $RUNS/.rig/config.reference.yaml instead, and config.sh then re-applies
  the recorded active config ($RUNS/.rig/active-config). config.yaml is never
  changed while ANY guarded run is live, whatever keys differ (the simple
  rule): the select refuses and says which runs. train without --config keeps
  whatever is active.
- Uploads go by rsync, not scp: rsync replaces a file by rename, so a
  guarded.sh that bash is still reading for a live run is not rewritten under
  it. swift runs as the admin user, who cannot see train's upload dir, so its
  script is fed on stdin (`ssh cc@... bash -s -- stop < swift.sh`).
- smoke waits for the run to end (about minutes + 2) and prints the gate;
  --no-wait returns after the launch. train --lr-reset waits (up to 30 min)
  for the first `rate x` line and prints it with the RESET/subjects/before
  lines.
- swift start --wait and eval score's judge use http://<host's address>:8080,
  polled from the Mac with urllib (no curl).
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path(__file__).resolve().parent / "remote"

SSH_OPTS = ["-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30"]

# cookbook 0.3's list, then what this tool adds (see the docstring)
SYNC_EXCLUDES = [".venv", "data", "data_*", "runs", ".git", ".claude",
                 "__pycache__", ".pytest_cache", ".codegraph",
                 "/weights/", "/weights-small/", ".minagi-dry-*",
                 "/config.yaml"]

FETCH_FILES = ["read.log", "samples.txt", "expert_rates.jsonl", "gpu.csv",
               "cmd", "config.yaml", "guard.log", "start", "end",
               "device_check.log"]

HARNESS = "/Volumes/stor/code/off/qwen/harness"
EGENT = "/Volumes/stor/code/egent"

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Rig:
    """The remote side: where things are, and how to reach them."""

    def __init__(self, a):
        self.host, self.admin = a.host, a.admin_host
        self.repo, self.runs, self.py = a.repo, a.runs.rstrip("/"), a.python
        self.dry = a.dry_run
        self.dir = f"{self.runs}/.rig"
        self.addr = self.host.rsplit("@", 1)[-1]
        self.uploaded = False
        self.shown = set()

    def run(self, cmd, stdin=None, capture=False):
        """Run (or, dry, print) one local command; its exit code, or its
        output too when capture is set."""
        line = shlex.join(cmd) + (f" < {stdin}" if stdin else "")
        if self.dry:
            print(f"$ {line}")
            return (0, "") if capture else 0
        f = open(stdin) if stdin else None
        try:
            p = subprocess.run(cmd, stdin=f, text=True,
                               stdout=subprocess.PIPE if capture else None)
        finally:
            if f:
                f.close()
        return (p.returncode, p.stdout) if capture else p.returncode

    def ssh(self, host, remote):
        return ["ssh", *SSH_OPTS, host, remote]

    def at(self, path):
        """host:path for rsync/scp; ~/x goes as x, which both read from home."""
        return f"{self.host}:{path[2:] if path.startswith('~/') else path}"

    def upload(self):
        """tools/rig/remote/ -> $RUNS/.rig/, once per invocation."""
        if not self.uploaded:
            self.uploaded = True
            return self.run(["rsync", "-a", f"{REMOTE}/", self.at(self.dir) + "/"])
        return 0

    def show(self, name, args):
        """Dry run: the script's text, with what each $N will be."""
        text = (REMOTE / name).read_text()
        deps = []
        if "lib.sh" in text:
            deps.append("lib.sh")
        if "launch_guarded" in text:
            deps.append("guarded.sh")
        for d in deps:
            if d not in self.shown:
                self.shown.add(d)
                self.show(d, [])
        print(f"# ---- {self.dir}/{name}")
        for i, v in enumerate(args, 1):
            print(f"#   ${i} = {v}")
        print(text.rstrip("\n"))
        print(f"# ---- end {name}")

    def script(self, name, *args, capture=False):
        """bash $RUNS/.rig/<name> <args> on the rig, with the paths in the
        environment lib.sh reads."""
        rc = self.upload()
        if rc:
            return (rc, "") if capture else rc
        args = [str(x) for x in args]
        repo = (f"~/{shlex.quote(self.repo[2:])}" if self.repo.startswith("~/")
                else shlex.quote(self.repo))   # the remote shell expands ~/
        remote = " ".join([f"RIG_REPO={repo}",
                           f"RIG_RUNS={shlex.quote(self.runs)}",
                           f"RIG_PY={shlex.quote(self.py)}",
                           "bash", shlex.quote(f"{self.dir}/{name}"),
                           *map(shlex.quote, args)])
        if self.dry:
            self.show(name, args)
        return self.run(self.ssh(self.host, remote), capture=capture)


# ---- argument helpers

def run_name(s):
    if not NAME.match(s):
        raise argparse.ArgumentTypeError(f"{s!r}: a run name is letters, digits, . _ -")
    return s


def number(s):
    """A number checked as one but passed on as typed (1e-4 stays 1e-4)."""
    try:
        float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r} is not a number")
    return s


def under(base, path):
    """path, or base/path when it is relative."""
    return path if path.startswith(("/", "~")) else f"{base}/{path}"


def mix_spec(s):
    """persona=0.5,chat=0.5 -> "persona:1,chat:1": folders per lane."""
    lanes = []
    for part in s.split(","):
        lane, _, w = part.partition("=")
        if not NAME.match(lane) or not w:
            raise argparse.ArgumentTypeError(f"--mix wants LANE=WEIGHT,..., got {part!r}")
        lanes.append((lane, float(number(w))))
    low = min(w for _, w in lanes)
    if low <= 0:
        raise argparse.ArgumentTypeError("--mix weights must be positive")
    counts = [(lane, w / low) for lane, w in lanes]
    if any(abs(c - round(c)) > 0.05 * c for _, c in counts) \
            or sum(round(c) for _, c in counts) > 12:
        raise argparse.ArgumentTypeError(
            f"--mix {s}: read shares equally per folder, so the weights must "
            "reduce to small whole ratios (at most 12 folders in all)")
    return ",".join(f"{lane}:{round(c)}" for lane, c in counts)


def model_spec(s):
    name, _, wd = s.partition("=")
    if not re.match(r"^[A-Za-z0-9_-]+$", name) or not wd:
        raise argparse.ArgumentTypeError(f"--model wants NAME=WEIGHTS_DIR, got {s!r}")
    return name, wd


def first_error(*codes):
    return next((c for c in codes if c), 0)


# ---- subcommands

def cmd_gpu_check(rig, a):
    return rig.script("gpu-check.sh", a.max_temp, a.max_used_mib)


def poll_swift(rig, timeout):
    url = f"http://{rig.addr}:8080/v1/models"
    if rig.dry:
        print(f"# poll GET {url} every 10 s until it lists a model "
              f"(up to {timeout} s)")
        return 0
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if b'"id"' in r.read():
                    print("swift-up")
                    return 0
        except OSError:
            pass
        time.sleep(10)
    print(f"strata-swift did not answer at {url} within {timeout} s")
    return 1


def cmd_swift(rig, a):
    if rig.dry:
        rig.show("swift.sh", [a.action])
    rc = rig.run(rig.ssh(rig.admin, f"bash -s -- {a.action}"),
                 stdin=str(REMOTE / "swift.sh"))
    if rc == 0 and a.action == "start" and a.wait:
        rc = poll_swift(rig, a.timeout)
    return rc


def cmd_sync(rig, a):
    code = rig.run(["rsync", "-a",
                    *[f"--exclude={x}" for x in SYNC_EXCLUDES],
                    f"{ROOT}/", rig.at(rig.repo) + "/"])
    if code:
        return code
    rc = rig.upload() or rig.run(
        ["rsync", "-a", str(ROOT / "config.yaml"),
         rig.at(f"{rig.dir}/config.reference.yaml")])
    if rc:
        return rc
    rc = rig.script("config.sh", *([a.config] if a.config else []))
    if a.test:
        rc = first_error(rc, rig.script("test.sh"))
    return rc


def cmd_smoke(rig, a):
    w = f"{rig.runs}/{a.name}"
    data = under(rig.runs, a.data) if a.data else f"{rig.runs}/w4-lanes/chat-train"
    held = under(rig.runs, a.held_out) if a.held_out else f"{rig.runs}/w4-lanes/chat-val"
    rc = rig.script("smoke.sh", w, a.config, a.dispatch, a.minutes, data, held)
    if rc or a.no_wait:
        return rc
    print(f"waiting for {a.name} to end (about {a.minutes} min plus startup)...")
    rig.script("wait.sh", w, "end", int(float(a.minutes)) + 30)
    return rig.script("status.sh", w)


def train_args(rig, a):
    """The train.py command line, as the run will get it."""
    w = f"{rig.runs}/{a.name}"
    args = ["train.py", "read", f"{w}/mix" if a.mix else a.data, "--save",
            "--weights-dir", f"{w}/weights", "--held-out", a.held_out,
            "--sample-every", a.sample_every, "--sample-log", f"{w}/samples.txt"]
    if a.minutes:
        args += ["--minutes", a.minutes]
    if a.max_chars:
        args += ["--max-chars", a.max_chars]
    if a.lr:
        args += ["--lr", a.lr]
    if a.lr_reset:
        args += ["--lr-reset"]
    if a.newborn_lr is not None:
        args += ["--newborn-lr" if a.newborn_lr else "--no-newborn-lr"]
    if a.newborn_half:
        args += ["--newborn-half", a.newborn_half]
    return args


def cmd_train(rig, a):
    if not (a.minutes or a.max_chars):
        print("train: give --minutes or --max-chars (an unattended run is bounded)")
        return 2
    w = f"{rig.runs}/{a.name}"
    rc = rig.script("train.sh", w, a.config or "-", a.dispatch,
                    under(rig.runs, a.from_weights) if a.from_weights else "-",
                    a.data if a.mix else "-", a.mix or "-",
                    *train_args(rig, a))
    if rc == 0 and a.lr_reset:
        print("waiting for the first `rate x` line (expect x1.000)...")
        rc = rig.script("wait.sh", w, "rate", 30)
    return rc


def cmd_status(rig, a):
    w = f"{rig.runs}/{a.name}"
    if rig.dry and a.watch:
        print(f"# repeats every {a.watch} s until the state line is not 'running'")
    while True:
        rc, out = rig.script("status.sh", w, capture=True)
        print(out, end="")
        if rig.dry or not a.watch or "\nstate: running" not in "\n" + out:
            return rc
        time.sleep(a.watch)
        print(f"\n---- {time.strftime('%H:%M:%S')}")


def cmd_eval(rig, a):
    e = f"{rig.runs}/{a.name}"
    if a.what == "gen":
        models = [f"{n}={under(rig.runs, wd)}" for n, wd in a.model]
        return rig.script("eval-gen.sh", e, a.shards, a.max_chars,
                          under(rig.runs, a.prompts), a.dispatch, *models)
    if a.what == "status":
        return rig.script("eval-status.sh", e)
    return eval_score(rig, a, e)


def eval_score(rig, a, e):
    """On the Mac: the replies back, shards concatenated per model, then
    tools/persona_score.py against the judge on the rig."""
    dest = Path(a.dest or ROOT / "runs" / a.name)
    if rig.dry:
        print(f"$ mkdir -p {shlex.quote(str(dest))}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
    rc = rig.run(["scp", "-q", rig.at(f"{e}/replies.*.jsonl"), f"{dest}/"])
    if rc:
        return rc
    shards = {}
    for f in dest.glob("replies.*.*.jsonl") if not rig.dry else []:
        m = re.match(r"^replies\.(.+)\.(\d+)\.jsonl$", f.name)
        if m:
            shards.setdefault(m.group(1), []).append((int(m.group(2)), f))
    if rig.dry:
        print(f"# concatenate {dest}/replies.<NAME>.<k>.jsonl in shard order "
              f"-> {dest}/<NAME>.jsonl")
        names = ["<NAME>"]
    else:
        names = sorted(shards)
        if not names:
            print(f"no replies.*.jsonl from {e}")
            return 1
        for n in names:
            with open(dest / f"{n}.jsonl", "w") as out:
                for _, f in sorted(shards[n]):
                    out.write(f.read_text())
            rows = sum(1 for _ in open(dest / f"{n}.jsonl"))
            print(f"{n} {rows}")
    cmd = [sys.executable, str(ROOT / "tools" / "persona_score.py"),
           "--harness", a.harness, "--egent", a.egent,
           "--replies", *[str(dest / f"{n}.jsonl") for n in names],
           "--names", *names,
           "--judge-url", f"http://{rig.addr}:8080",
           "--out-dir", str(dest / "score"), "--samples", str(a.samples)]
    if a.no_judge:
        cmd.append("--no-judge")
    return rig.run(cmd)


def cmd_fetch(rig, a):
    w = f"{rig.runs}/{a.name}"
    dest = Path(a.dest or ROOT / "runs" / a.name)
    if rig.dry:
        print(f"$ mkdir -p {shlex.quote(str(dest))}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
    for f in FETCH_FILES:
        # one scp per file, so a file the run never wrote (guard.log, end
        # while it runs) is skipped instead of failing the rest
        if rig.run(["scp", "-q", rig.at(f"{w}/{f}"), str(dest / f)]):
            print(f"skip {f} (not on the rig)")
    if not rig.dry:
        print(f"fetched into {dest}: {', '.join(sorted(p.name for p in dest.iterdir()))}")
    return 0


# ---- the parser

def parser():
    p = argparse.ArgumentParser(
        prog="rig.py", description=__doc__.split("\n\n")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The decisions behind the defaults are in this file's "
               "docstring (tools/rig/rig.py); usage: tools/rig/README.md.")
    p.add_argument("--host", default=os.environ.get("MINAGI_RIG_HOST", "train@10.0.1.19"),
                   help="the training user (default train@10.0.1.19, env MINAGI_RIG_HOST)")
    p.add_argument("--admin-host", default="cc@10.0.1.19",
                   help="the user with sudo, for strata-swift (default cc@10.0.1.19)")
    p.add_argument("--repo", default="~/train/mini-AGI", help="the rig's code (rsync copy)")
    p.add_argument("--runs", default="/mnt/data/runs/mini-agi", help="the rig's run dirs")
    p.add_argument("--python", default="/home/train/train/mini-AGI/.venv/bin/python")
    p.add_argument("--dry-run", action="store_true",
                   help="print every rsync/scp/ssh command and the remote scripts; run nothing")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("gpu-check", help="who holds the card; exit 1 unless it is free")
    s.add_argument("--max-temp", type=int, default=85)
    s.add_argument("--max-used-mib", type=int, default=1024)
    s.set_defaults(fn=cmd_gpu_check)

    s = sub.add_parser("swift", help="stop/start/status strata-swift (as --admin-host, sudo)")
    s.add_argument("action", choices=["stop", "start", "status"])
    s.add_argument("--wait", action="store_true",
                   help="start: poll :8080/v1/models until it answers")
    s.add_argument("--timeout", type=int, default=600, help="seconds --wait polls for")
    s.set_defaults(fn=cmd_swift)

    s = sub.add_parser("sync", help="rsync the code Mac -> rig, re-apply the run config")
    s.add_argument("--config", choices=["small", "reference"],
                   help="change the active run config (default: re-apply the recorded one)")
    s.add_argument("--test", action="store_true", help="then the CPU pytest suite on the rig")
    s.set_defaults(fn=cmd_sync)

    def dispatch(s):
        s.add_argument("--dispatch", choices=["padded", "exact"], default="padded",
                       help="MINAGI_DISPATCH (default padded, owner decision)")

    s = sub.add_parser("smoke", help="device_check + a short guarded read: the thermal/speed gate")
    s.add_argument("--name", type=run_name, default="w5-smoke")
    s.add_argument("--config", choices=["small", "reference"], default="small")
    dispatch(s)
    s.add_argument("--minutes", type=number, default="15")
    s.add_argument("--data", help="default $RUNS/w4-lanes/chat-train (relative: to --runs)")
    s.add_argument("--held-out", help="default $RUNS/w4-lanes/chat-val (relative: to --runs)")
    s.add_argument("--no-wait", action="store_true", help="return after the launch")
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("train", help="a guarded `train.py read --save` in a fresh run dir")
    s.add_argument("--name", type=run_name, required=True)
    s.add_argument("--data", required=True,
                   help="the folder to read, or with --mix the folder of lanes (relative: to --repo)")
    s.add_argument("--mix", type=mix_spec, help="LANE=WEIGHT,... e.g. persona=0.5,chat=0.5")
    s.add_argument("--held-out", required=True, help="relative: to --repo")
    s.add_argument("--config", choices=["small", "reference"],
                   help="default: keep the rig's active config")
    s.add_argument("--from-weights", help="cp -a into the fresh run dir (relative: to --runs)")
    s.add_argument("--lr", type=number)
    s.add_argument("--lr-reset", action="store_true",
                   help="full rate back (a checkpoint carries its own); waits for `rate x`")
    s.add_argument("--minutes", type=number)
    s.add_argument("--max-chars", type=number,
                   help="stop after this many characters (budget by characters)")
    s.add_argument("--sample-every", type=number, default="10")
    dispatch(s)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--newborn-lr", dest="newborn_lr", action="store_true", default=None)
    g.add_argument("--no-newborn-lr", dest="newborn_lr", action="store_false")
    s.add_argument("--newborn-half", type=number)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("status", help="temps, clocks, guard hits, speed, held-out per lane, ETA")
    s.add_argument("name", type=run_name)
    s.add_argument("--watch", type=int, metavar="SEC", help="repeat until the run ends")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("eval", help="persona eval: gen / status on the rig, score on the Mac")
    ev = s.add_subparsers(dest="what", required=True)
    e = ev.add_parser("gen", help="guarded persona_generate.py shards per model")
    e.add_argument("--name", type=run_name, required=True)
    e.add_argument("--model", type=model_spec, action="append", required=True,
                   metavar="NAME=WEIGHTS_DIR", help="repeatable; relative: to --runs")
    e.add_argument("--shards", type=int, choices=range(1, 11), default=3, metavar="1-10")
    e.add_argument("--max-chars", type=int, default=800)
    e.add_argument("--prompts", default="heldout.jsonl", help="relative: to --runs")
    dispatch(e)
    e = ev.add_parser("status", help="lines per shard, replies per file, guard hits")
    e.add_argument("--name", type=run_name, required=True)
    e = ev.add_parser("score", help="fetch replies, concatenate shards, persona_score.py")
    e.add_argument("--name", type=run_name, required=True)
    e.add_argument("--dest", help="default runs/<name> in this checkout")
    e.add_argument("--harness", default=HARNESS)
    e.add_argument("--egent", default=EGENT)
    e.add_argument("--samples", type=int, default=10)
    e.add_argument("--no-judge", action="store_true", help="structural check only, no Swift")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("fetch", help="copy a run's logs back to the Mac")
    s.add_argument("name", type=run_name)
    s.add_argument("--dest", help="default runs/<name> in this checkout")
    s.set_defaults(fn=cmd_fetch)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    return a.fn(Rig(a), a)


if __name__ == "__main__":
    sys.exit(main())
