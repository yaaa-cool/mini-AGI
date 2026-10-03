"""create.py's fallbacks must build a model halt_freeze can run."""

from dataclasses import fields

import torch

import minagi.config
from minagi.create import create
from minagi.recur import RecurConfig, RecurCoder

SMALL = dict(experts=2, resident=2, d_ff=16, depth=1, d_model=32,
             trunk_d_ff=64, n_head=2, block=32, top_k=1)


def build(d, **set_):
    names = {f.name for f in fields(RecurConfig)}
    cfg = RecurConfig(**{k: v for k, v in d.items() if k in names})
    for k, v in set_.items():
        setattr(cfg, k, v)
    return RecurCoder(cfg)


def test_fallbacks_agree_with_halt_freeze(tmp_path, monkeypatch):
    # a config that asks for halt_freeze but leaves the row shape unset
    monkeypatch.setattr(minagi.config, "load",
                        lambda path=None: {"model": {"halt_freeze": True}})
    d = create(str(tmp_path / "w"), verbose=False, **SMALL)
    assert (d["n_recur"], d["n_coda"], d["max_steps"]) == (1, 0, 24)
    m = build(d, halt_freeze=True, max_steps=3).eval()
    x = torch.randint(0, 265, (1, 8))
    with torch.no_grad():
        _, loss = m(x, x)
    assert torch.isfinite(loss)


def test_shipped_config_shape():
    c = minagi.config.load()
    assert minagi.config.get(c, "model.halt_freeze") is True
    assert minagi.config.get(c, "model.n_recur") == 1
    assert minagi.config.get(c, "model.n_coda") == 0
