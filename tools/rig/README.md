# tools/rig

Drive training on the Dell rig (RTX 3060, `train@10.0.1.19`) from the Mac,
from the repo root. Stdlib only; `--dry-run` prints every rsync/scp/ssh
command and the remote scripts it would run, and runs nothing.

```bash
python3 tools/rig/rig.py gpu-check [--max-temp 85] [--max-used-mib 1024]
python3 tools/rig/rig.py swift stop|start|status [--wait]
python3 tools/rig/rig.py sync [--config small|reference] [--test]
python3 tools/rig/rig.py smoke [--name w5-smoke] [--config small] [--dispatch padded|exact] [--minutes 15] [--no-wait]
python3 tools/rig/rig.py train --name N --data DIR --held-out DIR [--mix persona=0.5,chat=0.5]
    [--config small|reference] [--from-weights DIR] [--lr X] [--lr-reset]
    [--minutes M] [--max-chars N] [--sample-every 10] [--dispatch padded|exact]
    [--newborn-lr | --no-newborn-lr] [--newborn-half N]
python3 tools/rig/rig.py status N [--watch SEC]
python3 tools/rig/rig.py eval gen --name E --model NAME=WEIGHTS_DIR [--model ...] [--shards 3] [--max-chars 800]
python3 tools/rig/rig.py eval status --name E
python3 tools/rig/rig.py eval score --name E [--no-judge]
python3 tools/rig/rig.py fetch N [--dest runs/N]
```

Global flags go before the subcommand: `--host` (env `MINAGI_RIG_HOST`),
`--admin-host` (cc, for sudo), `--repo`, `--runs`, `--python`, `--dry-run`.

- Every GPU job (smoke, train, eval gen) runs under `MINAGI_DISPATCH=padded`
  unless `--dispatch exact`, starts with the GPU-free check, and runs under
  `guarded.sh` (SIGTERM at 85 C, `gpu.csv` every 10 s) in a fresh
  `$RUNS/<name>`, which holds `cmd` and `config.yaml` as run.
- `sync` keeps the rig's `config.yaml` and re-applies the active run config
  (`$RUNS/.rig/active-config`); config.yaml is never changed while a guarded
  run is live.
- `remote/` holds the rig-side scripts, uploaded to `$RUNS/.rig/` on each
  call. `w4-scripts/` is what W4 ran, kept as source material.

The decisions behind the defaults are in `rig.py`'s docstring. Tests:
`tests/test_rig_cli.py`.
