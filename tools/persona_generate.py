"""
Reply to the persona held-out prompts with a weights directory, learning off.

    python3 tools/persona_generate.py --weights runs/weights \
        --prompts heldout.jsonl --out replies.jsonl [--limit N] [--max-chars 800]

Each row of --prompts is {id, bucket, turns}, `turns` the user's turns. The
model answers them in order, each prompt the conversation so far - the user's
turns and the model's own earlier replies - rendered the way the persona lane
was written into training text (the harness's export_miniagi.py,
`render_dialog` and `escape_markers`), then an open `<bot>` tag:

    <user>\\n{turn}\\n</user>\\n<bot>\\n{reply}\\n</bot>\\n<user>\\n ... <bot>\\n

A reply ends at the first structural marker the model writes (`</bot>`, a
new `<user>`, `<|endoftext|>`, ...) or at --max-chars characters.

Decoding is serve.py's: temperature 0, the highest scoring character, with
the adaptation trace from config.yaml's decoding section. Nothing is drawn
from a random number generator unless pool.select_temperature is above 0;
--seed fixes that case.

Nothing is learned and nothing is written to --weights: the directory is
opened read-only, as `serve.py --no-learn` opens it, and no optimiser exists.
The architecture comes from the directory's manifest.json, not from
config.yaml; what config.yaml still supplies is run-time - pool.capacity_factor,
pool.select_temperature and the decoding trace - and is printed at the start.

One JSONL row per prompt row: {id, bucket, prompt, reply, chars, seconds},
plus `turns`, `replies` (one per user turn) and `stops`. `prompt` is the text
the last reply was generated from, `reply` the last reply, `chars` and
`seconds` the totals over all turns. Ids already in --out are skipped, so a
stopped run resumes.
"""

import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from minagi.tokenizer import SPECIAL_ID, ByteTokenizer  # noqa: E402

# harness training/persona/export_miniagi.py MARKERS, escape_markers and
# render_dialog, copied rather than imported: this runs where the model is,
# which need not be where the harness is
MARKERS = ["<think>", "</think>", "<user>", "</user>", "<bot>", "</bot>",
           "<g>", "</g>", "<|endoftext|>"]
_MARKER_RE = re.compile("|".join(re.escape(m) for m in MARKERS))
_TAG = {"user": "user", "assistant": "bot"}

# a reply ends at any marker that opens or closes a turn
STOP_IDS = {SPECIAL_ID[m] for m in ("</bot>", "<bot>", "<user>", "</user>",
                                   "<|endoftext|>")}


def escape_markers(text):
    return _MARKER_RE.sub(
        lambda m: m.group(0).replace("<", "&lt;").replace(">", "&gt;"), text)


def render_dialog(messages):
    out = []
    for m in messages:
        tag = _TAG.get(m.get("role"))
        val = (m.get("content") or "").strip()
        if tag is None or not val:
            continue
        out.append(f"<{tag}>\n{escape_markers(val)}\n</{tag}>\n")
    return "".join(out)


def build_prompt(messages):
    """The conversation so far, then the open tag that asks for a reply."""
    return render_dialog(messages) + "<bot>\n"


@torch.no_grad()
def reply_to(model, tok, prompt, max_chars, strength, decay):
    """
    Greedy continuation of `prompt` up to a turn marker or `max_chars`.

    serve.py's `stream` without the page: the prompt is read in chunks into
    fresh caches from position 0, then one forward per character; when the
    window fills, the newest half is re-read from position 0.
    """
    from minagi.decode import pick_next
    from minagi.precision import amp

    device = next(model.parameters()).device
    block = model.cfg.block
    ids = tok.encode(prompt).ids[-int(block * 0.9):]
    out = torch.tensor([ids], device=device)
    CHUNK = 512

    def read(text):
        caches = model.empty_caches()
        lg = None
        for i in range(0, text.shape[1], CHUNK):
            with amp(device):
                lg = model(text[:, i:i + CHUNK], caches=caches,
                           pos_offset=i)[0]
        return lg, caches

    logits, caches = read(out)
    pos = out.shape[1]
    produced, text, stop = [], "", "max_chars"
    while len(text) < max_chars:
        if logits is None:
            if pos + 1 > block:
                ctx = out[:, -(block // 2):]
                logits, caches = read(ctx)
                pos = ctx.shape[1]
            else:
                with amp(device):
                    logits = model(out[:, -1:], caches=caches,
                                   pos_offset=pos)[0]
                pos += 1
        nxt = pick_next(logits[:, -1, :].float(), out, temperature=0.0,
                        adapt_strength=strength, adapt_decay=decay)
        logits = None
        if int(nxt[0, 0]) in STOP_IDS:
            stop = "marker"
            break
        out = torch.cat([out, nxt], dim=1)
        produced.append(int(nxt[0, 0]))
        text = tok.decode(produced)
    return text[:max_chars].strip(), stop


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=800,
                    help="longest reply per turn, in characters")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N prompt rows only (0 = all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default=None,
                    help="compute dtype; default config.yaml "
                         "training.precision, else bf16 (as serve.py)")
    args = ap.parse_args()

    from minagi.config import get as _g, load as _lc
    from minagi.precision import set_compute_dtype
    from minagi.recur import load_any

    c = _lc()
    set_compute_dtype(args.precision or _g(c, "training.precision", "bf16"))
    strength = _g(c, "decoding.adapt_strength", 2.5)
    decay = _g(c, "decoding.adapt_decay", 0.88)
    torch.manual_seed(args.seed)

    dev = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # read_only: the pool marks nothing dirty, so nothing is written back
    model, _ = load_any(args.weights, dev, read_only=True)
    model.eval()
    tok = ByteTokenizer()
    pool = getattr(model, "pool", None)
    print(f"[loaded] {args.weights} on {dev}, block {model.cfg.block}; "
          f"decoding greedy, adapt {strength}/{decay}; select_temperature "
          f"{getattr(pool, 'select_temperature', 0.0)}, capacity_factor "
          f"{model.cfg.pool_capacity_factor} (config.yaml)", file=sys.stderr)

    with open(args.prompts, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass      # a line torn by a killed run is redone
    todo = [r for r in rows if r["id"] not in done]
    print(f"[prompts] {len(rows)} rows, {len(rows) - len(todo)} already in "
          f"{args.out}, {len(todo)} to do", file=sys.stderr)

    tot_chars, tot_secs = 0, 0.0
    with open(args.out, "a", encoding="utf-8") as fo:
        for n, row in enumerate(todo, 1):
            messages, replies, stops = [], [], []
            t0 = time.time()
            for turn in row["turns"]:
                messages.append({"role": "user", "content": turn})
                prompt = build_prompt(messages)
                reply, stop = reply_to(model, tok, prompt, args.max_chars,
                                       strength, decay)
                messages.append({"role": "assistant", "content": reply})
                replies.append(reply)
                stops.append(stop)
            secs = time.time() - t0
            chars = sum(len(r) for r in replies)
            tot_chars += chars
            tot_secs += secs
            fo.write(json.dumps({
                "id": row["id"], "bucket": row.get("bucket"),
                "prompt": prompt, "reply": replies[-1], "chars": chars,
                "seconds": round(secs, 3), "turns": row["turns"],
                "replies": replies, "stops": stops},
                ensure_ascii=False) + "\n")
            fo.flush()
            print(f"[{n}/{len(todo)}] {row['id']} {chars} chars "
                  f"{secs:.1f}s {stops}", file=sys.stderr)

    rate = tot_chars / tot_secs if tot_secs else 0.0
    print(f"[done] {len(todo)} rows, {tot_chars} chars in {tot_secs:.1f}s, "
          f"{rate:.1f} chars/s")


if __name__ == "__main__":
    main()
