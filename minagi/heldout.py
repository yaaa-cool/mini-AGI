"""
The level a held-out check reports at, and how the sample log records it.

A CHECK IS A SAMPLE. Every round scores a few blocks of each subject - a
fraction of a percent of what the subject holds - and a few blocks sit where
they sit: on the whole set they read a little above or below it, by an amount
that stays put for as long as the blocks do. So a session starts with a
larger evaluation spread over the whole set, reads the checks' own blocks
beside it, and reports every check that session at the larger evaluation's
level: what the check reads, plus the difference the two had at the start.
A check then moves with the model, at the level of an evaluation four times
its size.

THE LOG SAYS SO. The session banner carries one line for each evaluation the
start ran - the larger one, the checks' blocks, and the check as it was read
before checks were spread, from the top of each subject's files:

    # held-out at the start, spread over the whole set: 0.5975   arithmetic 0.6090   ...
    # held-out at the start, the checks' blocks: 0.6043   arithmetic 0.6120   ...
    # held-out at the start, from the top as checks read before: 0.6110   ...

and whatever draws or reports held-out reads them: a check logged before the
first of these banners was read from the top, and is moved onto the new level
by the difference that banner measured, subject by subject.

Pure Python, no torch, so the plotting tools can import it.
"""
import re

PREFIX = "# held-out at the start, "
SPREAD = "spread over the whole set"
CHECKS = "the checks' blocks"
TOP = "from the top as checks read before"

_LINE = re.compile(r"^# held-out at the start, ([^:\n]+): ([\d.]+)([^\n]*)$",
                   re.M)
_PAIR = re.compile(r"([a-z_]+) ([\d.]+)")


def banner(spread, checks=None, top=None):
    """The banner's lines for one session start, from dicts of subject ->
    loss. A dict left out or empty writes no line."""
    out = []
    for label, d in ((SPREAD, spread), (CHECKS, checks), (TOP, top)):
        if d:
            out.append(PREFIX + f"{label}: {sum(d.values()) / len(d):.4f}"
                       + "".join(f"   {k} {v:.4f}"
                                 for k, v in sorted(d.items())))
    return "".join(line + "\n" for line in out)


def levels(text):
    """Every session start's held-out lines, oldest first:
    [(offset of its first line, {label: {subject: loss}})]."""
    groups = []
    for m in _LINE.finditer(text):
        label = m.group(1).strip()
        if label == SPREAD or not groups:
            groups.append((m.start(), {}))
        groups[-1][1][label] = {k: float(v)
                                for k, v in _PAIR.findall(m.group(3))}
    return groups


def first_shift(text):
    """Where checks first reported at the spread level, and how far a check
    read from the top sat above that level there, subject by subject:
    (offset, {subject: top - spread}). None if the log never says."""
    for at, g in levels(text):
        if g.get(SPREAD) and g.get(TOP):
            return at, {k: g[TOP][k] - g[SPREAD][k]
                        for k in g[SPREAD] if k in g[TOP]}
    return None


def moved(val, dom, delta):
    """An old check (overall loss, {subject: loss} or None) on the spread
    level. The overall moves by the mean difference over the subjects the
    check itself has, so one from before a subject existed moves by its own
    subjects' share."""
    if not delta:
        return val, dom
    common = [k for k in (dom or {}) if k in delta] or list(delta)
    val = val - sum(delta[k] for k in common) / len(common)
    if dom:
        dom = {k: (v - delta[k] if k in delta else v) for k, v in dom.items()}
    return val, dom


def is_check(head):
    """Whether a log entry's header is a round's check. Only a check writes
    the rate controller's evidence line; the larger evaluation a session ends
    on does not."""
    return "evidence t " in head
