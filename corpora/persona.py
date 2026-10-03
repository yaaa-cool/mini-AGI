"""
Bring a persona dialog set in as a lane of its own.

    python3 -m corpora persona --src <export>

`<export>` is either a folder of text files already in the chat format - what
the persona harness's export_miniagi.py writes with --out-dir - or a .jsonl of
{"messages": [{"role": ..., "content": ...}, ...]} rows, which are rendered
here the way corpora/fetch.py renders a chat dataset. Either way the result is

    data/train/persona/NNNN/part-NNNNNN.txt
    data/val/persona/part-NNNNNN.txt

laid out like data/train/chat/hermes, so `train.py read data/train` reads
`persona` as one more subject and `--held-out data/val` scores it on its own.
The held-out files come from --src-val, or from the first --hold files of the
source when there is no separate held-out export.

It is not part of `corpora all`, because there is no persona to fetch: the
lane exists only when someone hands one over. Note what it costs - every
subject takes an equal share of the reading, so a ninth folder cuts the other
eight from 12.5% to 11.1% each, and a small persona folder is read many times
over. RIG-TRAINING.md has a replay recipe for adding it to a trained model.
"""

import argparse
import json
import os
import re
import shutil
import sys

from corpora.fetch import as_chat

# The structural markers minagi/tokenizer.py encodes as single ids wherever
# they occur in a file. One inside dialog content would forge a turn
# boundary, so the .jsonl path writes it as text, as export_miniagi.py does.
MARKERS = ["<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>",
           "<g>", "</g>", "<|endoftext|>"]
_MARKER_RE = re.compile("|".join(re.escape(m) for m in MARKERS))


def _escape(text):
    return _MARKER_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), text)


def _from_folder(src):
    """The export's text files, in order, one string apiece."""
    out = []
    for root, dirs, files in os.walk(src):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".txt"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    out.append(fh.read())
    return out


def _from_jsonl(src, group):
    """Dialogs rendered as turns and grouped into files, never split."""
    out, buf, skipped = [], [], 0
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = [dict(m, content=_escape(m.get("content") or ""))
                    for m in row.get("messages") or [] if isinstance(m, dict)]
            t = as_chat({"messages": msgs})
            if t is None:
                skipped += 1
                continue
            buf.append(t)
            if sum(len(x) for x in buf) >= group:
                out.append("".join(buf))
                buf.clear()
    if buf:
        out.append("".join(buf))
    if skipped:
        print(f"  {skipped:,} dialogs skipped (no user turn first, or no "
              f"bot turn)", flush=True)
    return out


def _load(src, group):
    return _from_folder(src) if os.path.isdir(src) else _from_jsonl(src, group)


def _write(texts, where, flat, shard):
    for i, text in enumerate(texts):
        sub = where if flat else os.path.join(where, f"{i // shard:04d}")
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, f"part-{i:06d}.txt"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(text)
    return sum(len(t) for t in texts)


def main():
    ap = argparse.ArgumentParser(prog="python3 -m corpora persona")
    ap.add_argument("--src", required=True,
                    help="export_miniagi.py's --out-dir folder, or a .jsonl "
                         "of {messages: [...]} rows")
    ap.add_argument("--src-val", default=None,
                    help="the held-out export, same forms as --src")
    ap.add_argument("--hold", type=int, default=0,
                    help="without --src-val, keep the first N files of --src "
                         "back as the held-out set")
    ap.add_argument("--out", default="data",
                    help="corpus root; writes <out>/train/persona and "
                         "<out>/val/persona")
    ap.add_argument("--group", type=int, default=16_000,
                    help="characters per file from a .jsonl (fetch.py --group)")
    ap.add_argument("--shard", type=int, default=2000,
                    help="files per NNNN/ folder (fetch.py --shard)")
    ap.add_argument("--force", action="store_true",
                    help="replace a persona lane that already has files")
    a = ap.parse_args()

    if a.hold and a.src_val:
        ap.error("--hold and --src-val both name the held-out set; give one")
    train_dir = os.path.join(a.out, "train", "persona")
    val_dir = os.path.join(a.out, "val", "persona")
    for d in (train_dir, val_dir):
        if os.path.isdir(d) and os.listdir(d):
            if not a.force:
                print(f"{d} already has files - --force to replace it",
                      file=sys.stderr)
                return 1
            # emptied, not merged into: a shorter rebuild would otherwise
            # leave the tail of the old one behind
            shutil.rmtree(d)

    train = _load(a.src, a.group)
    if a.src_val:
        val = _load(a.src_val, a.group)
    else:
        hold = min(a.hold, len(train) // 10) if a.hold else 0
        val, train = train[:hold], train[hold:]
    if not train or not any("<bot>" in t for t in train):
        print(f"no chat text in {a.src}", file=sys.stderr)
        return 1

    chars = _write(train, train_dir, False, a.shard)
    print(f"  train/persona  {len(train):>5} files  {chars/1e6:>7.2f}M chars")
    if val:
        chars = _write(val, val_dir, True, a.shard)
        print(f"  val/persona    {len(val):>5} files  {chars/1e6:>7.2f}M chars")
    else:
        print("  no held-out set: --src-val or --hold, so a run reports "
              "persona loss on its own", file=sys.stderr)
    print(f"  train:  python3 train.py read {a.out}/train --save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
