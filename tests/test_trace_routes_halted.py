"""read --trace-routes crashed once some characters had halted and others ran on."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from minagi.pool import capture_routes  # noqa: E402
from minagi.recur import RecurConfig, RecurCoder  # noqa: E402


def _model():
    torch.manual_seed(0)
    cfg = RecurConfig(vocab_size=265, d_model=32, n_head=2, d_ff=64,
                      n_prelude=1, n_recur=1, n_coda=0, max_steps=3,
                      block=32, use_pool=True, pool_experts=4, pool_d_ff=8,
                      pool_top_k=2, halt_freeze=True, halt_thresh=0.5)
    m = RecurCoder(cfg).train()
    readout = m._readout

    def halves(y, targets):
        # every other character halts after the first pass, the rest run on
        logits, lam, ce = readout(y, targets)
        lam = torch.full_like(lam, 0.1)
        lam[:, ::2] = 0.9
        return logits, lam, ce
    m._readout = halves
    return m


def test_partial_halt_traces_and_writes(tmp_path):
    m = _model()
    x = torch.randint(0, 265, (1, 16))
    _, plain = m(x, x)
    with capture_routes() as got:
        _, loss = m(x, x)
    assert torch.equal(plain, loss)            # observing changes nothing
    assert len(got) == 3
    for ids, w in got:                         # every call covers every character
        assert ids.shape == w.shape == (16, 2)
    for ids, w in got[1:]:                     # the halted ones route nowhere
        assert (ids[::2] == -1).all() and (w[::2] == 0).all()
        assert (ids[1::2] >= 0).all()
    assert (got[0][0] >= 0).all()

    # a shorter, shallower second chunk has to stack with the first
    m.sample_depth = lambda: 2
    with capture_routes() as got2:
        m(x[:, :12], x[:, :12])
    lane, pool = SimpleNamespace(name="s"), m.pool
    pool.slots = [0, 1]
    tr = train._Tracer(str(tmp_path / "trace.npz"), 2, swap_chunks=0)
    tr.add(got, pool, lane, 0, 0, loss.item(), 16, x[:, -8:])
    tr.add(got2, pool, lane, 0, 1, loss.item(), 28, x[:, -8:])
    tr.write(m.cfg, "fp32", 8)
    z = np.load(tmp_path / "trace.npz")
    assert z["ids"].shape == z["weights"].shape == (2, 3, 16, 2)
    meta = json.loads(str(z["meta"]))["chunks"]
    assert [(c["calls"], c["window"]) for c in meta] == [(3, 16), (2, 12)]
    # the short window is padded before its first character, the missing
    # pass after its last call
    assert (z["ids"][1, :, :4] == -1).all() and (z["ids"][1, 2] == -1).all()
    assert np.array_equal(z["ids"][0], np.stack([g[0] for g in got]))
