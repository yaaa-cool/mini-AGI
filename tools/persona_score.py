"""
Score persona replies from several models with the harness's own checks.

    python3 tools/persona_score.py --harness <harness> --egent <egent> \
        --replies b.jsonl author.jsonl a.jsonl --names B author A \
        --out-dir runs/persona-eval [--judge-url http://10.0.1.19:8080] \
        [--judge-model <id>] [--samples 10] [--no-judge]

The replies are tools/persona_generate.py's output. Each one is checked the
way the harness filtered the persona training data, by importing its
training/persona/filter.py rather than copying it:

  structural   egent's structural persona check (structural_check.ts through
               bun) over every assistant turn; a reply passes when no turn is
               rewritten
  judge        the teacher grades the whole dialog 1-10 against
               judge_rubric.md; a reply passes at --min-score (7, as filter.py)
               or above. Judgements are cached per model in
               --out-dir/judge_cache.<name>.jsonl, so a rerun asks only for
               what changed or failed.

Only ids present in every --replies file are scored, so the models are
compared on the same prompts. Writes --out-dir/scored.<name>.jsonl per model
and --out-dir/summary.md: a side-by-side table, a per-bucket breakdown and
--samples prompts with every model's reply and verdicts, chosen with --seed
round-robin across buckets.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def messages_of(row):
    """The dialog as filter.py's rows hold it: user turns and the replies."""
    out = []
    for turn, reply in zip(row["turns"], row["replies"]):
        out.append({"role": "user", "content": turn})
        out.append({"role": "assistant", "content": reply})
    return out


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harness", required=True)
    ap.add_argument("--egent", required=True)
    ap.add_argument("--replies", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--judge-url", default="http://10.0.1.19:8080",
                    help="OpenAI-compatible server; /v1 is added if missing")
    ap.add_argument("--judge-model", default="swift-1.5-iq3_xxs")
    ap.add_argument("--judge-no-thinking", action="store_true")
    ap.add_argument("--min-score", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    if len(args.names) != len(args.replies):
        ap.error("--names needs one name per --replies file")

    sys.path.insert(0, str(Path(args.harness).resolve()))
    from training.persona import filter as pf

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_model = [{r["id"]: r for r in read_jsonl(p)} for p in args.replies]
    ids = sorted(set.intersection(*(set(m) for m in by_model)))
    for name, m in zip(args.names, by_model):
        print(f"[replies] {name}: {len(m)} rows, {len(m) - len(ids)} not "
              f"in every file", file=sys.stderr)
    if not ids:
        raise SystemExit("no id is in every --replies file")

    # filter.py keys a row by (seed_id, cand); cand is the model's index
    rows = [[{"seed_id": i, "cand": mi, "messages": messages_of(m[i])}
             for i in ids] for mi, m in enumerate(by_model)]

    verdicts = pf.structural_verdicts([r for rs in rows for r in rs],
                                      Path(args.egent))

    judged = [{} for _ in by_model]          # id -> cache record
    jstats = [None for _ in by_model]
    if not args.no_judge:
        rubric = pf.RUBRIC_PATH.read_text(encoding="utf-8")
        url = args.judge_url.rstrip("/")
        url = url if url.endswith("/v1") else url + "/v1"
        client = pf.JudgeClient(url, args.judge_model,
                                no_thinking=args.judge_no_thinking)
        for mi, name in enumerate(args.names):
            cache = out_dir / f"judge_cache.{name}.jsonl"
            _, jstats[mi] = pf.judge_all(rows[mi], client, rubric, cache,
                                         args.concurrency)
            print(f"[judge] {name}: {jstats[mi]}", file=sys.stderr)
            # the cache's last record per key is the judgement in effect
            judged[mi] = {k[0]: rec
                          for k, rec in pf.load_judge_cache(cache).items()}

    scored = []
    for mi, (name, m) in enumerate(zip(args.names, by_model)):
        recs = []
        for i in ids:
            r = m[i]
            reasons = verdicts[(i, mi)]
            rec = judged[mi].get(i) or {}
            score = rec.get("score") if not args.no_judge else None
            s_ok = not reasons
            j_ok = None if args.no_judge else (score or 0) >= args.min_score
            recs.append({
                "id": i, "bucket": r.get("bucket"), "turns": r["turns"],
                "replies": r["replies"], "chars": r["chars"],
                "structural_pass": s_ok, "structural_reasons": reasons,
                "judge_score": score, "judge_reasons": rec.get("reasons"),
                "judge_pass": j_ok,
                "both_pass": None if j_ok is None else (s_ok and j_ok)})
        with open(out_dir / f"scored.{name}.jsonl", "w",
                  encoding="utf-8") as f:
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        scored.append(recs)

    def stats(recs):
        n = len(recs)
        s = sum(x["structural_pass"] for x in recs)
        chars = sum(x["chars"] for x in recs) / n if n else 0.0
        if args.no_judge:
            return [str(n), pct(s, n), "-", "-", "-", f"{chars:.0f}"]
        j = sum(bool(x["judge_pass"]) for x in recs)
        b = sum(bool(x["both_pass"]) for x in recs)
        sc = [x["judge_score"] for x in recs if x["judge_score"] is not None]
        mean = f"{sum(sc) / len(sc):.2f} ({len(sc)} scored)" if sc else "-"
        return [str(n), pct(s, n), pct(j, n), mean, pct(b, n), f"{chars:.0f}"]

    def table(head, body):
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return lines

    md = ["# Persona eval", "",
          f"{len(ids)} prompts present in every replies file. Structural: "
          "egent's structural persona check (harness filter.py "
          "`structural_verdicts`). Judge: "
          + ("not run (--no-judge)." if args.no_judge else
             f"`{args.judge_model}` at {args.judge_url} against "
             f"judge_rubric.md, pass at score >= {args.min_score}; an "
             "unscored reply fails."),
          "", "## Models", ""]
    md += table(["model", "n", "structural pass", "judge pass",
                 "mean judge score", "both pass", "mean reply chars"],
                [[name] + stats(recs)
                 for name, recs in zip(args.names, scored)])
    md.append("")
    for name, st, path in zip(args.names, jstats, args.replies):
        md.append(f"- {name}: `{path}`" + (f", judge {st}" if st else ""))

    buckets = sorted({x["bucket"] or "" for x in scored[0]})
    md += ["", "## Per bucket", "",
           "Each cell is structural / judge / both pass rate."
           if not args.no_judge else "Each cell is the structural pass rate.",
           ""]
    body = []
    for b in buckets:
        cells = [b, str(sum(1 for x in scored[0] if (x["bucket"] or "") == b))]
        for recs in scored:
            sub = [x for x in recs if (x["bucket"] or "") == b]
            st = stats(sub)
            cells.append(st[1] if args.no_judge
                         else f"{st[1]} / {st[2]} / {st[4]}")
        body.append(cells)
    md += table(["bucket", "n"] + args.names, body)

    # samples: round-robin over buckets, a seeded draw within each
    rng = random.Random(args.seed)
    pools = {b: [i for i, x in zip(ids, scored[0]) if (x["bucket"] or "") == b]
             for b in buckets}
    for b in buckets:
        rng.shuffle(pools[b])
    picks = []
    while len(picks) < min(args.samples, len(ids)):
        for b in buckets:
            if pools[b] and len(picks) < args.samples:
                picks.append(pools[b].pop())
    index = [{x["id"]: x for x in recs} for recs in scored]
    md += ["", "## Samples", "",
           f"{len(picks)} prompts, seed {args.seed}, round-robin across "
           "buckets.", ""]
    for k, i in enumerate(picks, 1):
        first = index[0][i]
        md += [f"### {k}. `{i}` ({first['bucket']})", ""]
        for t, turn in enumerate(first["turns"], 1):
            md += [f"**User{f' turn {t}' if len(first['turns']) > 1 else ''}"
                   ":**", "", "> " + turn.strip().replace("\n", "\n> "), ""]
        for name, ix in zip(args.names, index):
            x = ix[i]
            v = ("structural " + ("pass" if x["structural_pass"] else
                                  "FAIL " + "; ".join(x["structural_reasons"])))
            if not args.no_judge:
                v += (f", judge {x['judge_score']}"
                      f" ({'pass' if x['judge_pass'] else 'fail'})"
                      f": {x['judge_reasons'] or ''}")
            md += [f"**{name}** - {v}", ""]
            for t, rep in enumerate(x["replies"], 1):
                lab = f"turn {t}: " if len(x["replies"]) > 1 else ""
                md += ["> " + lab + (rep.strip() or "(empty)")
                       .replace("\n", "\n> "), ""]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n",
                                        encoding="utf-8")
    print(f"[done] {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
