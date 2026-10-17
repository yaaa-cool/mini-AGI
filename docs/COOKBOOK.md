# mini-AGI rig cookbook: run the small training runs yourself

This is the W4 cookbook (2026-10-08), rewritten around the rig CLI in
`tools/rig/` and updated for upstream 574a397 (merged as 165fe32). Every step
gives the `rig.py` command first, then what it runs underneath on the rig, and,
where it helps, the command W4 actually ran as a worked example.

- CLI usage: `tools/rig/README.md`. The decisions behind the defaults are in
  the module docstring of `tools/rig/rig.py`.
- Run every command from the repo root on the Mac. Global flags go BEFORE the
  subcommand. `python3 tools/rig/rig.py --dry-run <sub> ...` prints every
  rsync, scp and ssh command and the text of each remote script, and runs
  nothing. Use it to see exactly what a step will do.
- **W4 figures** (speeds, temperatures, losses, timings) are labelled as such.
  They were measured on 2026-10-08, before 574a397, and the held-out losses
  among them are on the old scale (see "Held-out scale" in section 4).

Tags on each step:

- **[read-only]**: looks at the rig and changes nothing. Allowed without a
  release.
- **[owner release]**: changes rig state (code, services, config) but does not
  train.
- **[GPU - owner release]**: loads a model on the GPU. It needs the owner's
  release for that specific run (RIG-TRAINING.md, top).

Rig: `train@10.0.1.19`, RTX 3060 12 GB. Code lives in `~/train/mini-AGI`, an
rsync copy and not a git checkout, with the venv in `.venv`. Run dirs are
`/mnt/data/runs/mini-agi/<name>` (`$RUNS` below; `--runs` changes it). The
CLI's own scripts are uploaded to `$RUNS/.rig/` on every call, by rsync, so a
script a live run is still reading is replaced by rename and not rewritten
under it.

Shorthand in the by-hand blocks and W4 worked examples below, as W4 used it
on the rig (log in as `train@10.0.1.19`, then `cd ~/train/mini-AGI`):

```bash
R=/mnt/data/runs/mini-agi                  # = $RUNS
P=/home/train/train/mini-AGI/.venv/bin/python
L=$R/w4-lanes                              # the chat-only lanes (0.6)
W=$R/<run name>                            # the run dir
```

What a run dir holds: `guarded.sh` writes `read.log` (training output),
`gpu.csv` (nvidia-smi every 10 s), `pid`, `start`, `end` (the exit code and
time, written when the run exits) and `guard.log` (only if the 85 C guard
fired). Training adds `samples.txt`, `expert_rates.jsonl` and `weights/`.
`launch_guarded` in `tools/rig/remote/lib.sh` adds `cmd` (the exact command,
with its `MINAGI_DISPATCH`) and `config.yaml` (the config it ran with).

---

## 0. Prereqs

### 0.1 Is the GPU free? [read-only]

```bash
python3 tools/rig/rig.py gpu-check
python3 tools/rig/rig.py gpu-check --max-temp 85 --max-used-mib 1024   # the defaults, spelled out
```

Underneath: `tools/rig/remote/gpu-check.sh` prints the hostname, the
nvidia-smi summary (name, temperature, SM clock, memory, power limit,
sw_thermal), the compute apps holding VRAM, whether strata-swift is active,
the active run config with its `d_model` line, and any live guarded runs. It
then runs `gpu_free` (`lib.sh`), the same check every GPU job runs first, and
exits 1 if more than 1024 MiB of VRAM is in use, any compute app is running,
or the card is already at 85 C.

What to look at: about 11.6 GB used means strata-swift (the llama server on
:8080) is loaded, so stop it first (0.2). W4 figure: idle after the cooling
service read 46 C at SM 405 MHz. Leave the 100 W power limit alone.

### 0.2 Stop and start strata-swift [owner release]

```bash
python3 tools/rig/rig.py swift stop
python3 tools/rig/rig.py swift start --wait
python3 tools/rig/rig.py swift status
```

Underneath: `tools/rig/remote/swift.sh`, run as the admin user (`--admin-host`,
default `cc@10.0.1.19`) and fed on stdin (`ssh cc@... bash -s -- stop <
swift.sh`), because `cc` cannot see train's upload dir. `stop` runs `sudo
systemctl stop strata-swift`, waits 5 s, and prints the service state and the
VRAM in use. `start` runs `sudo systemctl start` and waits 3 s. `--wait` then
polls `http://10.0.1.19:8080/v1/models` from the Mac with urllib until Swift
answers (`--timeout` sets how long).

Worked example (W4, recorded):

```bash
ssh cc@10.0.1.19 'sudo systemctl stop strata-swift && sleep 5 && systemctl is-active strata-swift; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader'
```

What to look at: `inactive` and memory near 0 after a stop, `active` after a
start. Start Swift again when you finish: claude-r and the eval judge use it.

### 0.3 Sync the code from the Mac to the rig [owner release]

```bash
python3 tools/rig/rig.py sync --test
```

Underneath, in order:

1. `rsync -a` of the repo to `train@10.0.1.19:train/mini-AGI/`, excluding
   W4's list (`.venv data data_* runs .git .claude __pycache__ .pytest_cache
   .codegraph`) plus `/weights/`, `/weights-small/`, `.minagi-dry-*` and
   `/config.yaml`. The upstream weights snapshot is gitignored and published
   separately, so the code needs neither weights dir, and a local weights dir
   must never land on the rig's checkpoints.
2. The Mac's `config.yaml` goes to `$RUNS/.rig/config.reference.yaml`.
3. `tools/rig/remote/config.sh` re-applies the recorded active config
   (`$RUNS/.rig/active-config`), so the rig's `config.yaml` is never reset by
   a sync.
4. With `--test`, `tools/rig/remote/test.sh` runs the CPU suite on the rig
   (`CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests`, last 5
   lines) and exits with pytest's code.

Worked example (W4, recorded; this copied `config.yaml` over the rig's, which
is the gotcha the CLI now handles):

```bash
rsync -a --exclude .venv --exclude data --exclude 'data_*' --exclude runs --exclude .git --exclude .claude --exclude __pycache__ --exclude .pytest_cache --exclude .codegraph ./ train@10.0.1.19:train/mini-AGI/ && ssh train@10.0.1.19 'cd ~/train/mini-AGI && CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests | tail -1'
```

What to look at: the pytest summary, all passed. W4 saw 25 passed and the perf
wave 29; the count has grown since, so judge by "passed" and no failures.

### 0.4 config.small.yaml against config.yaml [owner release]

```bash
python3 tools/rig/rig.py sync --config small        # select SMALL for smoke, A and the persona run
python3 tools/rig/rig.py sync --config reference    # select REFERENCE for the author weights (B)
```

`minagi/config.py` reads only `<repo>/config.yaml`; there is no `--config`
flag in `train.py`. config.small cuts width only: d_model 512 to 384, expert
width 2048 to 1024, top-k 8 to 4, 16 resident out of 32 experts, about 42M
params against 209M+. Depth and halting are left as shipped.

Underneath: after the code sync, `config.sh small` calls `select_config`
(`lib.sh`), which copies `config.small.yaml` (or the reference copy from 0.3)
over `config.yaml`, records the choice in `$RUNS/.rig/active-config`, and
prints the `d_model` line (384 is small, 512 is reference). It refuses, and
names the runs, if `config.yaml` would change while ANY guarded run is live.
You rarely need this step by itself: `smoke` and `train` take `--config` and
select it the same way, and `train` without `--config` keeps whatever is
active. `gpu-check` prints the active config.

Worked example (W4, recorded, by hand on the rig):

```bash
cp -n config.yaml /mnt/data/runs/mini-agi/config.reference.yaml   # backup, once
cp config.small.yaml config.yaml && cmp config.yaml config.small.yaml
grep -m1 "d_model" config.yaml
```

**Since 574a397 (open owner decision):** both configs carry the newborn keys
(`training.newborn_lr: true`, `newborn_half: 15000`). config.small.yaml still
keeps `capacity_factor: 1.5` and `balance: 2.5e-4`, where upstream's
config.yaml moved to `capacity_factor: 0` and `balance: 3.5e-5`. Whether
config.small follows upstream is not decided.

### 0.5 The rig-side scripts

There is nothing to install by hand. Every call uploads `tools/rig/remote/` to
`$RUNS/.rig/` and runs `ssh train@10.0.1.19 bash $RUNS/.rig/<script> <args>`.
`guarded.sh` is W4's script with the 85 C limit named once (`GUARD_C`): it
runs the command with output to `$W/read.log`, samples nvidia-smi every 10 s
into `$W/gpu.csv`, and sends SIGTERM at 85 C or above. W4's originals
(`guarded.sh`, `status.sh`, `genC.sh`, `smoke.sh`, `runA.sh`, `runB2.sh` and
the rest) are kept verbatim in `tools/rig/w4-scripts/` as source material.

### 0.6 Chat-only lane folders (one-time, already on the rig) [owner release]

There is no CLI step for this; the folders exist from W4 and are the default
data for `smoke`. Recorded W4 command, to rebuild them on the rig:

```bash
L=/mnt/data/runs/mini-agi/w4-lanes
mkdir -p $L/chat-train $L/chat-val
cp -rs /home/train/train/mini-AGI/data/train/chat $L/chat-train/chat   # symlink tree, nothing moved
cp -rs /home/train/train/mini-AGI/data/val/chat   $L/chat-val/chat
```

`read` gives every top-level subfolder an equal share, so a folder that
contains only `chat/` means 100% chat.

---

## 1. Smoke run: the thermal and speed gate, 15 min [GPU - owner release]

```bash
python3 tools/rig/rig.py swift stop
python3 tools/rig/rig.py smoke --name w5-smoke
```

Defaults: `--config small`, `--dispatch padded`, `--minutes 15`, data
`$RUNS/w4-lanes/chat-train`, held-out `$RUNS/w4-lanes/chat-val`. `smoke` waits
for the run to end (about minutes + 2) and prints the gate; `--no-wait`
returns after the launch. `--dispatch exact` reproduces W4's dispatch.

Underneath: `tools/rig/remote/smoke.sh` runs `gpu_free`, refuses an existing
run dir (`fresh_dir`), selects the config, exports `MINAGI_DISPATCH`, and
launches under `guarded.sh`:

```bash
$PY tools/device_check.py --window 2048 --chunk 2048 > $W/device_check.log 2>&1
$PY train.py read $DATA --save --weights-dir $W/weights --held-out $HELD --minutes 15 --sample-every 5 --sample-log $W/samples.txt
```

Then `tools/rig/remote/wait.sh $W end 45` blocks until `$W/end` exists, and
`tools/rig/remote/status.sh $W` prints the gate.

What to look at:

- **Gate:** SM clock at 1500 MHz or more under load, `swThermalActive=0`, and
  `maxT` well under 85. If the clock sits below 1500 or sw_thermal goes
  Active, stop there.
- **Speed:** the `speed (latest)` line, which is `reading N char/s` from the
  last `samples ->` line.
- **VRAM:** `vram NNNNMB` among the gate lines, or `mem_used_mib` in gpu.csv.
- **Fallback:** `device_check.log` names the attention kernel and bf16 or fp32.
- **Newborn cost:** compare chars/s with RIG-TRAINING.md §6 (7,215 chars/s
  padded single-run). See section 4 for why it may be lower now.

W4 figures (exact dispatch, before 574a397): max 62 C, SM 1500, 5,300 char/s,
peak VRAM 1.4 GB, GPU util only 13-33%. At 42M params the model is bound by
kernel launches and the CPU, which is why padded dispatch helps. Held-out chat
went 5.52 to 2.66 nats in 15 min (old scale).

---

## 2. Run A: from scratch, chat only, config.small, about 3 h [GPU - owner release]

```bash
python3 tools/rig/rig.py train --name w5-A --config small --data /mnt/data/runs/mini-agi/w4-lanes/chat-train --held-out /mnt/data/runs/mini-agi/w4-lanes/chat-val --minutes 180 --sample-every 10
```

Underneath: `tools/rig/remote/train.sh` runs `gpu_free`, refuses an existing
`$RUNS/w5-A`, selects config.small, exports `MINAGI_DISPATCH=padded`, and
launches under `guarded.sh`:

```bash
train.py read /mnt/data/runs/mini-agi/w4-lanes/chat-train --save --weights-dir $RUNS/w5-A/weights --held-out /mnt/data/runs/mini-agi/w4-lanes/chat-val --sample-every 10 --sample-log $RUNS/w5-A/samples.txt --minutes 180
```

`--data` and `--held-out` are relative to the rig's repo (train.py's working
directory), so the lane folders are given as absolute paths. `train` refuses
to launch without `--minutes` or `--max-chars`.

Worked example (W4, recorded `runA.sh`, exact dispatch):

```bash
PYTHONUNBUFFERED=1 nohup $R/guarded.sh $W $P train.py read $L/chat-train --save --weights-dir $W/weights --held-out $L/chat-val --minutes 180 --sample-every 10 --sample-log $W/samples.txt > $W/guard.out 2>&1 &
```

What to look at, in `read.log` order:

- **`before:`** for a random model sits near ln(256) = 5.55, so it is
  guessing uniformly over bytes. Since 574a397 the next line says what the
  checks read (`checks read N blocks of each ...`) and where the old
  from-the-top reading sits.
- **Startup lines:** model size (`total 42.29M` in W4), `rate x1.000`,
  `1 subjects ... chat`, and the newborn line (`newborn experts start at the
  full rate and come down ...`) unless `--no-newborn-lr`.
- **Every 10 min, a `samples ->` line:** train and held-out loss, `reading N
  char/s`, `capacity dropped X% of routing`, and the elapsed minutes.
- **Pool growth:** `pool N experts (+1)`; `weights/ checkpointed at step S`
  marks a checkpoint.
- **End:** `after: X ... delta` is held-out after the run and its change.

W4 figures (exact dispatch, old held-out scale): 28.5M chars in 182 min,
held-out chat 5.52 to 1.88. That was 4.8k char/s alone and about 3.3k while B
ran alongside. It is not fluent, and that is expected. With padded, the perf
wave measured about 7.2k char/s single-run, so expect about 1.5x the
characters in the same 180 min, less whatever the newborn rule costs.

---

## 3. Run B: continue the author's weights on persona + chat, about 2 h

### 3.1 The author weights (one-time, already done) [owner release]

No CLI step. W4 downloaded them to `$RUNS/author/hf` (3.4 GB); recorded, on the
rig:

```bash
mkdir -p $R/author && cd $R/author && nohup $P -c "from huggingface_hub import snapshot_download; snapshot_download(\"Volotat/mini-AGI-undertrained\", local_dir=\"/mnt/data/runs/mini-agi/author/hf\")" > download.log 2>&1 &
```

### 3.2 Check that they load in the synced code, on the CPU [owner release]

No CLI step. Recorded W4 command, on the rig in `~/train/mini-AGI`:

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python tools/persona_generate.py --weights $R/author/hf/weights --prompts $R/heldout.jsonl --out $R/author/loadcheck.jsonl --limit 1 --max-chars 120 --device cpu 2>&1 | tail -6
```

What to look at: one reply row and no traceback. Worth repeating after an
upstream merge that touches the model, as 574a397 did.

### 3.3 The training folder: persona 1/2 + chat 1/2

`data/replay-chat` exists on the rig from W4 (persona 15.0M chars, chat slice
15.0M, hard links). Instead of rebuilding it by hand, `train --mix` builds the
same thing inside the run dir:

```bash
python3 tools/rig/rig.py train --name w5-B --config reference --from-weights author/hf/weights --data data/train --mix persona=0.5,chat=0.5 --held-out data/val --lr 1e-4 --lr-reset --minutes 120 --sample-every 10
```

Underneath: `build_mix` in `tools/rig/remote/train.sh` makes
`$RUNS/w5-B/mix/{persona,chat}`. The first lane is linked whole; every other
folder gets files up to the size of one of the first lane's folders, in
replay.sh's fixed shuffled order. Hard links where the corpus shares the run
dir's disk, symbolic links otherwise. Weights must reduce to small whole
ratios (0.5,0.5 is 1:1; 0.25,0.75 is 1:3; at most 12 folders), because `read`
gives every top-level folder an equal share. For the gentler 1/9 mix in
RIG-TRAINING.md §4, read `data/replay` (8 subjects plus persona).

### 3.4 Launch [GPU - owner release]

```bash
python3 tools/rig/rig.py swift stop
python3 tools/rig/rig.py train --name w5-B --config reference --from-weights author/hf/weights --data data/replay-chat --held-out data/val --lr 1e-4 --lr-reset --minutes 120 --sample-every 10
```

To reproduce W4 exactly, add `--dispatch exact`. Padded was measured on
config.small only; on the 209M reference model it is unmeasured.

Underneath: `train.sh` runs `gpu_free`, refuses an existing `$RUNS/w5-B`,
selects the reference config (refused only if that would change config.yaml
while a guarded run is live), `cp -a`s
`$RUNS/author/hf/weights` to `$RUNS/w5-B/weights` (`--from-weights` is
relative to `$RUNS`), and launches:

```bash
train.py read data/replay-chat --save --weights-dir $RUNS/w5-B/weights --held-out data/val --sample-every 10 --sample-log $RUNS/w5-B/samples.txt --minutes 120 --lr 1e-4 --lr-reset
```

Because of `--lr-reset`, `rig.py` then runs `tools/rig/remote/wait.sh $W rate
30`: it waits up to 30 min for the first `rate x` line and prints W4's
`waitB2.sh` filter (the `before`, `rate x`, `subjects` and `RESET` lines).

Worked example (W4, recorded `runB2.sh`, the good relaunch):

```bash
cp $R/config.reference.yaml config.yaml; grep -m1 "d_model" config.yaml
W=$R/w4-B-lrreset; mkdir -p $W && cp -a $R/author/hf/weights $W/weights
PYTHONUNBUFFERED=1 nohup $R/guarded.sh $W $P train.py read data/replay-chat --save --weights-dir $W/weights --held-out data/val --lr 1e-4 --lr-reset --minutes 120 --sample-every 10 --sample-log $W/samples.txt > $W/guard.out 2>&1 &
```

What to look at in the lines `train --lr-reset` prints:

- `plasticity RESET: full learning rate restored`, then `rate x1.000`.
- `2 subjects ... chat, persona`.
- `before: X on data/val`. This is this run's own baseline on the new scale.
  W4's `before: 0.6664` was on the old from-the-top scale and is not
  comparable; do not gate on it.

**The lr gotcha.** The plasticity controller's rate is saved inside the
weights. The author's checkpoint carries `rate x0.119`, so `--lr 1e-4` alone
really ran at about 1.2e-5. W4's first B launch (`w4-B`) did just that; it was
stopped with `kill -TERM $(cat $W/pid)` after 0.05M chars and relaunched on a
fresh copy with `--lr-reset`. Use `--lr-reset` whenever you continue someone
else's or an old checkpoint on new data (without it the controller only
resets itself once held-out shows the ground has moved). It sets the main
scale to x1; newborn
experts (section 4) start at the full `--lr` either way.

W4 figures (exact dispatch, old held-out scale): 5.67M chars in 120 min
(1.15k char/s, 6.6 GB VRAM). Held-out over all 9 lanes went 0.666 to 0.778,
+0.111 of forgetting. Persona held-out 1.011 to 1.055 did NOT fall, and
self-knowledge went 0.284 to 0.789.

---

## 4. Monitoring while it trains [read-only]

```bash
python3 tools/rig/rig.py status w5-A               # once
python3 tools/rig/rig.py status w5-A --watch 300   # every 5 min until the run ends
python3 tools/rig/rig.py fetch w5-A                # logs back to runs/w5-A on the Mac
```

Underneath: `tools/rig/remote/status.sh $RUNS/w5-A` prints the last gpu.csv
row, then `maxT meanT meanSM minSM swThermalActive hwThermalActive`, guard
hits, the latest speed, the gate lines from read.log, the held-out banner and
the latest check (see below), the start time and ETA (from `--minutes` in
`cmd`, or for a `--max-chars` run an estimate from the latest check's
characters and reading rate), and a last line `state: running|ended|dead`.
`--watch` repeats it until the state is no longer `running`, which replaces W4's `until ssh ...
test -f end` loop. `fetch` copies `read.log`, `samples.txt`,
`expert_rates.jsonl`, `gpu.csv`, `cmd`, `config.yaml`, `guard.log`, `start`,
`end` and `device_check.log` with scp into `runs/<name>` of the checkout you
run it from (`--dest` changes it); run it from the main checkout.

By hand on the rig, W4's recorded greps are still useful:

```bash
grep 'samples ->' $W/read.log | cut -c1-200       # one line per sample interval
grep -n -A1 '^held-out loss' $W/samples.txt       # held-out per LANE at every check
kill -TERM $(cat $W/pid)                          # stop early (no CLI step); the 85 C guard does the same
                                                  # --save checkpoints every save_every minutes and at the end
```

How to read the numbers:

- **Loss units:** cross-entropy in nats per character. bits/char = nats /
  0.693, and perplexity = e^nats (1.88 nats is perplexity 6.6). A random model
  sits at about 5.5.
- **Held-out scale (since 574a397).** `before`, `after` and every check read
  whole blocks spread over each held-out subject, instead of reading from the
  top. Each session writes a banner to samples.txt; its `# held-out at the
  start, spread over the whole set: ...` line is the start's evaluation, and
  every check reports at that level, with a `level +X` field giving how far
  the check's own blocks sit from it. All W4 held-out figures in this
  cookbook are on the old from-the-top scale and are NOT comparable with new
  runs. Gate a new run on its own `before:` line and banner, never on a W4
  number. `status` prints the banner line and the latest check side by side.
- **Overfitting:** the check's header line ends `gap X`, which is held-out
  minus train. A gap growing positive means memorising.
- **Per-lane line** (under the header): `arithmetic 0.640 chat 0.740 ...
  persona 1.055 ...` (W4 values). For persona work, this is the number that
  matters.
- **Forgetting:** the held-out set includes lanes the run is not reading. If
  they rise, that is forgetting. `read` prints it at the end as `after: X
  +/-e +delta`. A lane is fine if it moves by less than the printed +/-, which
  is the measurement noise.
- **Speed:** `reading N char/s` is the speed while learning. The final "over
  the whole session" figure includes evals and samples, so it runs about half
  that.
- **Learning rate controller (574a397).** At every check it eases the rate
  down, by at most 0.4%, unless held-out is clearly improving; it rises only
  on clear improvement and never above `--lr`; the floor is x0.01. After a
  restart (a new session) the first 5 readings are held to re-level, so the
  rate does not move on them. `--lr-reset` puts the main scale back at x1.
- **Per-expert learning rate (574a397).** Under `training.newborn_lr` (on by
  default in both configs), an expert born during the read starts at the full
  `--lr` and decays to the main rate: every `newborn_half` (15000) of its own
  steps halves how far above the main rate it is, in log terms. `train` passes `--newborn-lr`, `--no-newborn-lr` or
  `--newborn-half N` through to `train.py`. Each expert's rate at every check,
  as a factor of `lr`, goes to `expert_rates.jsonl` next to the sample log.
  It costs one GPU wait per step, which is unmeasured on the rig: if chars/s
  drops against RIG-TRAINING.md §6, rerun with `--no-newborn-lr`.
- **`--drop-quietest N`** is interactive (it waits for confirmation on stdin
  before deleting experts) and `rig.py` does not expose it. Never use it in an
  unattended run.

---

## 5. Eval C: 300 held-out persona prompts, replies, then a score

The order matters: generation needs the GPU free (Swift stopped), and the
judge needs Swift running.

### 5.1 Prompts onto the rig (one-time, already at `$RUNS/heldout.jsonl`) [owner release]

No CLI step; `eval gen --prompts` defaults to this file. Recorded:

```bash
cd /Volumes/stor/code/off/qwen/harness && scp -q training/persona/data/heldout.jsonl train@10.0.1.19:/mnt/data/runs/mini-agi/heldout.jsonl
```

### 5.2 Generate: guarded shards per model [GPU - owner release]

```bash
python3 tools/rig/rig.py swift stop
python3 tools/rig/rig.py eval gen --name w5-C --model author=author/hf/weights --model B=w5-B/weights --model A=w5-A/weights
python3 tools/rig/rig.py eval status --name w5-C                                      # [read-only] progress
```

Defaults: `--shards 3`, `--max-chars 800`, `--prompts $RUNS/heldout.jsonl`,
`--dispatch padded`. Model weights are relative to `$RUNS`.

Underneath: `tools/rig/remote/eval-gen.sh` (W4's `genC.sh` for any number of
models) runs one `gpu_free`, checks every weights dir exists, refuses any
existing `gen-<NAME>-<k>` dir, splits the prompts into `$RUNS/w5-C/prompts.<k>.jsonl`
once (and refuses a later call with a different shard count), then launches
one guarded `tools/persona_generate.py --weights WD --prompts prompts.<k>.jsonl
--out replies.<NAME>.<k>.jsonl --max-chars 800` per model and shard. It does
not touch config.yaml: the model's shape comes from the manifest in the
weights folder, and config.yaml only supplies defaults such as
`training.precision` (W4 generated A with the reference config active and it
worked). `tools/rig/remote/eval-status.sh` (W4's `genstat.sh`) prints the card,
lines per shard, replies per file, guard hits and `shards ended: N of M`.

Worked example (W4, recorded):

```bash
bash $R/genC.sh author $R/author/hf/weights && bash $R/genC.sh B $R/w4-B-lrreset/weights
bash $R/genC.sh A $R/w4-A/weights
```

W4 figures: greedy decoding at about 6 char/s per process; 900 replies (3
models x 300) took about 85 min with `--max-chars 400`. The 400 cap cut 40% of
the author's turns, 34% of B's and 11% of A's, which is why the default is now
800; expect generation to take longer than W4's 85 min (unmeasured). Padded
dispatch on the reference-sized author and B models is unmeasured; add
`--dispatch exact` to match W4.

### 5.3 Start Swift for the judge [owner release]

```bash
python3 tools/rig/rig.py eval status --name w5-C     # wait for "shards ended: 9 of 9"
python3 tools/rig/rig.py swift start --wait
```

`--wait` replaces W4's separate poll of `http://10.0.1.19:8080/v1/models`.

### 5.4 Pull the replies and score on the Mac [owner release: uses Swift on the rig]

```bash
python3 tools/rig/rig.py eval score --name w5-C
python3 tools/rig/rig.py eval score --name w5-C --no-judge     # structural check only, no Swift needed
```

Underneath (`eval_score` in `rig.py`): scp of `$RUNS/w5-C/replies.*.jsonl` to
`runs/w5-C/` on the Mac, the shards concatenated per model in shard order into
`runs/w5-C/<NAME>.jsonl` (it prints the row count of each), then:

```bash
python3 tools/persona_score.py --harness /Volumes/stor/code/off/qwen/harness --egent /Volumes/stor/code/egent --replies runs/w5-C/A.jsonl runs/w5-C/B.jsonl runs/w5-C/author.jsonl --names A B author --judge-url http://10.0.1.19:8080 --out-dir runs/w5-C/score --samples 10
```

Each model should show 300 rows. Judgements are cached in
`score/judge_cache.<name>.jsonl`, so a rerun asks only for what is missing.
W4 figure: the judge took about 2 h for 900 dialogs at concurrency 4.

### 5.5 Reading `score/summary.md`

- **Models table:** structural pass, judge pass (score 7 or higher out of
  10), mean judge score, both pass, and mean reply chars.
- **Buckets:** a per-bucket breakdown, then 10 sample prompts, each with
  every model's reply and verdicts.
- **Structural (egent's check):** catches canned openers and links. It was
  about 100% for every W4 model, and that says nothing, because incoherent
  text has neither.
- **Judge (Swift against `judge_rubric.md`):** W4 scored 0% pass for all
  three models, with means of 1.00 to 1.05. The rubric grades persona
  qualities such as honesty and tone; on gibberish everything scores 1, so it
  cannot tell the models apart. Use the judge only once samples.txt replies
  read as sentences. Until then, the gate is held-out loss (section 6).

---

## 6. Next: persona run on top of Run A's chat weights (NOT YET RUN) [GPU - owner release]

The plan from the W4 DONE plus the owner's padded decision: start from A's
weights (config.small, already chat-trained), lr 3e-5, the full persona lane
(15.0M chars) at half of the reading mixed with chat, padded dispatch, and a
character budget instead of minutes.

One-time, on the rig: a held-out folder with chat and persona only (W4 sketch,
reconstructed; no CLI step):

```bash
L=/mnt/data/runs/mini-agi/w5-lanes; mkdir -p $L/val
cp -rs /home/train/train/mini-AGI/data/val/chat    $L/val/chat
cp -rs /home/train/train/mini-AGI/data/val/persona $L/val/persona
```

Then:

```bash
python3 tools/rig/rig.py swift stop
python3 tools/rig/rig.py train --name w5-persona --config small --from-weights w4-A/weights --data data/replay-chat --held-out /mnt/data/runs/mini-agi/w5-lanes/val --lr 3e-5 --lr-reset --max-chars 30000000 --sample-every 10
python3 tools/rig/rig.py status w5-persona --watch 600
```

Underneath: `train.sh` selects config.small, `cp -a`s `$RUNS/w4-A/weights` to
`$RUNS/w5-persona/weights` (w4-A stays untouched), and launches, padded:

```bash
train.py read data/replay-chat --save --weights-dir $RUNS/w5-persona/weights --held-out /mnt/data/runs/mini-agi/w5-lanes/val --sample-every 10 --sample-log $RUNS/w5-persona/samples.txt --max-chars 30000000 --lr 3e-5 --lr-reset
```

then waits for and prints the `rate x` lines: expect `rate x1.000` and `2
subjects`. `--mix persona=0.5,chat=0.5 --data data/train` in place of `--data
data/replay-chat` builds the same half/half folder fresh inside the run dir
(3.3).

- **Budget:** `--max-chars` stops after that many characters. For a half/half
  mix, about 2x the persona lane's characters reads persona once, so
  30,000,000 for the 15.0M lane. Confirm with the final `read N characters`
  line. Prefer it to `--minutes`: the speed with padded dispatch and the
  newborn rule on this run is unmeasured. W4 sizing for reference: A read
  28.5M chars in 182 min with exact dispatch, and padded measured about 1.5x.
  `status` shows an estimated ETA for a `--max-chars` run once the first
  check has landed.
- **Newborn rule:** on by default. If the speed line is well below RIG-TRAINING.md
  §6, stop and rerun on a fresh name with `--no-newborn-lr`.

Gate (from samples.txt, `grep -A1 '^held-out loss'`, or `status`):

- **Persona held-out falls:** this run's banner (start) value against its
  final check. This is the pass/fail. In W4's B it did not fall.
- **Chat stays within the printed +/-:** a chat rise means forgetting, so
  lower the lr or give persona a smaller share.
- **The persona-vs-chat gap** (persona loss minus chat loss, per check) should
  shrink. A at the start has persona far above chat, because it has never
  seen persona.
- **Eval C, only if both pass:** section 5 on this model (`--model
  P=w5-persona/weights`; 800 chars is the default), and look at the samples
  before believing the judge.

---

## 7. Gotchas (from W4 PROGRESS/DONE and the perf wave, updated for the CLI)

- **The rig reads the config file, not a flag.** Still true: `config.yaml` IS
  the config, and some keys are re-read mid-run. Handled by the CLI:
  `select_config` refuses to change `config.yaml` while ANY guarded run is
  live, whatever keys differ (A and B in W4 ran side by side only because the
  re-read keys agreed). Each run dir keeps the `config.yaml` it ran with, and
  `gpu-check` prints the active one.
- **A code rsync reset `config.yaml` to the reference config.** Handled by
  the CLI: `sync` excludes `config.yaml`, ships the Mac's copy to
  `$RUNS/.rig/config.reference.yaml`, and re-applies the active config. By
  hand, the old rsync still has this problem.
- **A reused `--weights-dir` resumes instead of starting fresh.** Handled by
  the CLI: `smoke`, `train` and `eval gen` refuse an existing dir, and
  `--from-weights` is always a fresh `cp -a`. There is no resume through the
  tool.
- **lr x0.119: a checkpoint brings its controller rate with it.** Still true.
  Use `--lr-reset`; `train` then waits for and prints the `rate x` line, so
  confirm `rate x1.000` there.
- **`read` without `--save` is a dry read** and keeps nothing. Handled:
  `smoke` and `train` always pass `--save`. Still true for a hand-run `read`.
- **Unbounded runs.** New: `train` refuses to launch without `--minutes` or
  `--max-chars`, so nothing unattended reads the whole corpus by accident.
- **Runs side by side share the GPU.** W4 figure: A dropped from 4.8k to 3.3k
  char/s while B ran alongside. Through the CLI this no longer happens by
  accident: every GPU job starts with `gpu_free`, which refuses while any
  compute app runs.
- **Padded dispatch** is the default for every GPU job (owner decision);
  `--dispatch exact` opts out, and `cmd` records which. About 1.5x on
  config.small and statistically equivalent to exact (-0.35% loss, CI -1.6 to
  +0.9%), but ponder ran slightly higher at all 4 seeds, so keep an eye on
  it. Neither mode is bitwise reproducible on this card. Unmeasured on the
  reference model.
- **Temperature guard.** `guarded.sh` sends SIGTERM at 85 C (`GUARD_C`), and
  `gpu_free` refuses to start a job on a card already at `--max-temp`. A guard
  hit shows in `status` as `guard hits: N`.
- **strata-swift holds about 11.6 GB.** Handled: `swift stop|start` runs as
  `cc` with sudo, and `gpu_free` names Swift when VRAM is held. You still have
  to stop it before training and start it for the judge.
- **Auto-mode denials (agents only).** Still true: the classifier denied the
  rsync into `~/train/mini-AGI` ("Modify Shared Resources") and stopping Swift
  as `train` ("Remote Shell Writes"). That is why W4's agents used `cc` and
  the owner ran the rsync, and why W4 never used the `~/train/mini-AGI-small`
  checkout it created.
- **The 400-char generation cap.** Handled: `eval gen` defaults to
  `--max-chars 800`.
- **Structural ~100% is vacuous, and the judge is useless on incoherent
  replies.** Still true. At this scale, gate on held-out loss.
- **Held-out scale changed in 574a397.** New: W4's held-out figures are on the
  old from-the-top scale. Compare a run only with its own `before:` and
  banner.
- **Newborn per-expert lr costs a GPU wait per step**, unmeasured. New: watch
  chars/s against RIG-TRAINING.md §6 and use `--no-newborn-lr` if it drops.
- **`--drop-quietest` is interactive.** New: never in an unattended run; the
  CLI does not expose it.
- **config.small vs upstream config.yaml.** New, open owner decision:
  config.small keeps `capacity_factor 1.5` and `balance 2.5e-4`, where
  upstream's config.yaml moved to 0 and 3.5e-5.
- **The performance CPU governor** was applied by the owner at about 15:15 UTC
  during W4. The exact command was not recorded (reconstructed: `sudo cpupower
  frequency-set -g performance`). Not part of the CLI.
- **Leftovers on the rig as of W4** (not deleted; the owner's call):
  `~/train/mini-AGI-small`, `$RUNS/w4-B` (the aborted first B), `data/replay`
  (hard links), `~/train/mini-AGI/{minagi/fastpath.py,tests/test_fastpath.py,tools/test_perf_tools.py}`,
  and `$RUNS/perf2` (about 0.5 GB). A `sync` does not delete files on the rig,
  so the stray code files stay until removed by hand.
- **W4 outputs:** on the rig in `$RUNS/{w4-smoke,w4-A,w4-B-lrreset,w4-C,author/hf}`;
  Mac copies in `runs/w4/{A,B,eval-C}` of the main checkout.
