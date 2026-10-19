# tools/rig

Drive training on the Dell rig (two RTX 3060 12 GB cards, `train@10.0.1.19`)
from the Mac, from the repo root. Stdlib only; `--dry-run` prints every
rsync/scp/ssh command and the remote scripts it would run, and runs nothing.

```bash
python3 tools/rig/rig.py gpu-check [--gpu N] [--max-temp 85] [--max-used-mib 1024]
python3 tools/rig/rig.py swift stop|start|status [--wait]
python3 tools/rig/rig.py sync [--config small|reference] [--test]
python3 tools/rig/rig.py smoke [--name w5-smoke] [--config small] [--dispatch padded|exact] [--gpu N|auto]
    [--minutes 15] [--no-wait]
python3 tools/rig/rig.py train --name N --data DIR --held-out DIR [--mix persona=0.5,chat=0.5]
    [--config small|reference] [--from-weights DIR] [--lr X] [--lr-reset]
    [--minutes M] [--max-chars N] [--sample-every 10] [--dispatch padded|exact]
    [--newborn-lr | --no-newborn-lr] [--newborn-half N] [--gpu N|auto]
python3 tools/rig/rig.py status N [--watch SEC]
python3 tools/rig/rig.py eval gen --name E --model NAME=WEIGHTS_DIR [--model ...] [--shards 3] [--max-chars 800]
    [--gpu N,M|auto]
python3 tools/rig/rig.py eval status --name E
python3 tools/rig/rig.py eval score --name E [--no-judge]
python3 tools/rig/rig.py fetch N [--dest runs/N]
```

Global flags go before the subcommand: `--host` (env `MINAGI_RIG_HOST`),
`--admin-host` (cc, for sudo), `--repo`, `--runs`, `--python`, `--dry-run`.

- Every GPU job (smoke, train, eval gen) takes `--dispatch padded|exact`,
  default `padded`, and runs under that `MINAGI_DISPATCH` (written into the
  run's `cmd` either way). It starts with the GPU-free check and runs under
  `guarded.sh` (SIGTERM at 85 C, `gpu.csv` every 10 s) in a fresh
  `$RUNS/<name>`, which holds `gpu` (the card), `cmd` and `config.yaml` as
  run.
- Two cards, one run per card. `--gpu N` pins a job to nvidia-smi's card N
  (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N`, so the job's
  `cuda:0` is that card); `--gpu auto`, the default, takes the first free
  card and fails if none is. The free check (VRAM, compute apps, live guarded
  runs, temperature), the 85 C guard and `gpu.csv` are all per card; a run's
  guard fires on its own card only. Two jobs side by side:
  `train --name a ... --gpu 0` and `train --name b ... --gpu 1`. They share
  one checkout and so one `config.yaml`: the second must use the active
  config (a `--config` that would change it is refused while the first is
  live). The cards share the case, so `status` prints both temperatures.
- `eval gen --gpu 0,1` spreads its jobs (models times shards) round-robin
  over the cards; `auto`, the default, uses every free card.
- `gpu-check` reports every card (or `--gpu N`) with its compute apps and a
  free/busy verdict, and exits 0 if a card checked is free.
- Relative paths resolve differently per subcommand: **train**'s `--data`
  and `--held-out` against `--repo` (train.py's working directory), but
  **smoke**'s `--data` and `--held-out` against `--runs` (its defaults are
  `$RUNS/w4-lanes/chat-train` and `chat-val`). `--from-weights` and
  `eval gen --model` weights resolve against `--runs`. A `~/` path is the
  rig's home.
- `sync` keeps the rig's `config.yaml` and re-applies the active run config
  (`$RUNS/.rig/active-config`); config.yaml is never changed while a guarded
  run is live. Before the first select, the active config is read from
  config.yaml's `d_model` (384 small, 512 reference); any other width makes
  the sync stop and ask for `--config small|reference`.
- `status` gives an ETA from `--minutes`, and for a `--max-chars` run an
  estimate from the characters read at the last check and its reading rate.
- `remote/` holds the rig-side scripts, uploaded to `$RUNS/.rig/` on each
  call. `w4-scripts/` is what W4 ran, kept as source material.

The decisions behind the defaults are in `rig.py`'s docstring. Tests:
`tests/test_rig_cli.py`.
