# Training on the Dell rig

`train@10.0.1.19`, RTX 3060 12 GB. Repo `~/train/mini-AGI` (venv `.venv`),
run outputs under `/mnt/data/runs/mini-agi`. **Owner release.** No rig work starts until the owner has released the rig.
Every command marked **[GPU - owner release]** loads the model on the GPU and
needs that release for the specific run. Corpus builds and tests are CPU-only.
Never touch other GPU processes on the box.

Experiment config: `config.small.yaml` (a companion change adds it); code reads
`config.yaml` by default, so select it as its header describes.

Session setup: `ssh train@10.0.1.19`, `cd ~/train/mini-AGI && . .venv/bin/activate`,
`W=/mnt/data/runs/mini-agi/<run>` (one folder per run).

## 1. Corpus (CPU)

```bash
# keep the corpus on the data disk (once, if data/ does not exist yet)
mkdir -p /mnt/data/runs/mini-agi/data && ln -s /mnt/data/runs/mini-agi/data data
python3 -m corpora all --limit 5000       # a trial slice, minutes
python3 -m corpora all                    # the sampled default, a few GB
ls data/train   # wikipedia stories chat reasoning arithmetic code chess self-knowledge
```

A lane on disk is skipped; `all --only <lane> --force` rebuilds it.
`self-knowledge` lands in `data/{train,val}/self-knowledge/` (fix #10), where
`serve.py` primes from; its numbers come from `MINAGI_FACTS_WEIGHTS=$W/weights`
(that model) or `=config` (a fresh one).

## 2. Persona lane (CPU)

`corpora persona` takes the harness export folder or the raw `.jsonl`:

```bash
# on the Mac, in the harness repo
python -m training.persona.export_miniagi --in training/persona/data/sft.jsonl \
    --out-dir /tmp/persona --held-out /tmp/persona-val --hold 20
rsync -a /tmp/persona /tmp/persona-val train@10.0.1.19:/mnt/data/runs/mini-agi/export/
# on the rig
python3 -m corpora persona --src /mnt/data/runs/mini-agi/export/persona \
    --src-val /mnt/data/runs/mini-agi/export/persona-val
# or straight from the .jsonl, holding the first 20 files back:
python3 -m corpora persona --src sft.jsonl --hold 20
```

Writes `data/train/persona/NNNN/part-NNNNNN.txt` and `data/val/persona/`;
`--force` replaces an existing lane. Not part of `corpora all`; once built, a
plain `read data/train` reads it as a ninth subject.

## 3. Training from scratch [GPU - owner release]

```bash
mkdir -p $W
nohup python3 train.py read data/train --save --weights-dir $W/weights \
    --held-out data/val --sample-every 10 --sample-log $W/samples.txt \
    > $W/read.log 2>&1 &     # --save: checkpoint every save_every min + at end
```

## 4. Adding persona to a trained model (replay mix)

Reading a new subject alone for 1M characters cost +0.13 nats on the other
lanes; the same subject read as one more lane cost nothing (README,
continual-learning section). So do not read `data/train/persona` alone: read a
folder that mixes it with slices of every prior lane.

`read` gives every top-level folder an EQUAL share of the reading, whatever its
size (`corpora/build.py` docstring, `_lanes` in `train.py`). The ratio is
therefore set by how many subjects the folder has, not by file counts:

```bash
R=data/replay; mkdir -p $R && cp -al data/train/persona $R/persona   # hard links
budget=$(du -sb data/train/persona | cut -f1)   # each slice about persona's size
for lane in wikipedia stories chat reasoning arithmetic code chess self-knowledge; do
  mkdir -p $R/$lane
  find data/train/$lane -type f | shuf --random-source=<(yes) | while read -r f; do
    [ "$(du -sb $R/$lane | cut -f1)" -ge "$budget" ] && break
    ln "$f" "$R/$lane/$(echo "${f#data/train/$lane/}" | tr / _)"
  done
done
```

- **Ratio.** Nine subjects puts persona at 1/9 (about 11%), the corpus
  convention, and is the suggested start. For about 20%, nest the slices in
  four subjects (`$R/prior-1/{wikipedia,stories}`, ...) for 1/5. Above 1/3 a
  small persona lane is re-read so often it is memorised.
- **Slice size.** A slice at least as large as the persona lane is never
  re-read more often than persona itself.

```bash
cp -a $W/weights $W/weights-persona   # a COPY: paging can write expert files
                                      # even on a dry read (upstream issue #6)
# [GPU - owner release] dry read: held-out per subject before and after;
# rerun with --save to keep it
python3 train.py read data/replay --weights-dir $W/weights-persona \
    --held-out data/val --lr 1e-4 --minutes 30
```

- **lr.** With a config, `read` defaults to `training.lr` (3e-4), not the 5e-5
  the README names. Start an add-on read at 1e-4 (unmeasured starting point);
  `minagi/plasticity.py` then moves it with held-out. `trunk_lr_mult` (0.1) keeps the shared trunk slow.
- **Pass.** persona held-out falls. Each of the other eight stays within the
  +/- that `read` prints. A lane that rises past that is forgetting: lower
  `--lr`, or give persona a smaller share.

## 5. Forgetting check [GPU - owner release]

`replication/forgetting_probe.py` reads one subject massed (or, with
`--rotate`, interleaved as the control) and scores every held-out domain at
intervals. Give each run its own fresh copy of the weights:

```bash
cp -a $W/weights-persona $W/probe-w
python3 replication/forgetting_probe.py --weights $W/probe-w \
    --domain persona --read-root data/train --held-out data/val \
    --steps 256 --at 0,64,128,256 --chunk 2048 --lr 1e-4 \
    --trunk-lr-mult 0.1 --out $W/probe/persona-massed.json
# control: same flags on a new copy, plus --out $W/probe/persona-rotate.json
#   --rotate persona,wikipedia,stories,chat,reasoning,arithmetic,code,chess
jq -r '.domains[] as $d | "\($d) \(.rows[-1][$d] - .baseline[$d])"' \
    $W/probe/persona-massed.json            # loss change per domain
```

Damage in the massed run that the rotated run lacks is the forgetting the
replay mix in section 4 avoids.

## Tests (CPU)

```bash
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests/
```

## SYNC

Local `main` is our integration branch. `upstream/main` mirrors
github.com/volotat/mini-AGI: never commit to it, and never create a local
branch that shadows it. Our upstream fixes are commits titled `fix #N: ...`.

- **Sync:** `scripts/sync-upstream.sh` (refuses unless on a clean `main`, and
  never pushes), or by hand: `git fetch upstream && git merge upstream/main`.
  Always a merge, never a rebase, because `main` only moves forward.
- **Our patches:** `git log --oneline upstream/main..main --grep '^fix #'`.
- **Is a patch redundant?** Any one of these: the issue is closed upstream;
  upstream has touched the same code (`git log upstream/main -- <files>`); or
  our regression test passes on a clean `upstream/main` checkout without the
  patch.
- **Conflict with an upstream fix:** take upstream's version and drop ours
  (`git checkout --theirs <file>`, then `git add`, `git commit`). The script
  lists, for each conflicted file, which `fix #N` commits touch it.
- **Dropping a redundant patch outside a conflict:** `git revert <our fix
  commit>` on `main`. After any of these, rerun the tests on the rig.
