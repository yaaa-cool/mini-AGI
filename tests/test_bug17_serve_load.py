"""#17: serve.py must start on a weights directory that was never saved."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import serve  # noqa: E402
from minagi.create import create  # noqa: E402


def test_serve_starts_on_never_saved_dir(tmp_path, monkeypatch):
    src = str(tmp_path / "weights")
    create(src, verbose=False, experts=3, resident=1, d_ff=8, depth=1,
           d_model=8, trunk_d_ff=16, n_head=1, block=64, max_steps=1, top_k=1)
    with open(os.path.join(src, "manifest.json")) as f:
        assert not json.load(f).get("paged")    # the case that crashed
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(sys, "argv", [
        "serve.py", "--weights", src, "--device", "cpu", "--no-learn",
        "--prime-chars", "0", "--precision", "fp32"])
    ran = []
    monkeypatch.setattr(serve.app, "run", lambda **kw: ran.append(kw))
    serve.main()                    # the startup banner raised AttributeError
    assert ran
    # loaded through the paging path, so the expert panel has slots to show
    assert hasattr(serve.STATE["model"].pool, "slots")


def test_saved_in_memory_dir_keeps_its_load_path(tmp_path):
    """A `train.py stream` save has experts/ too, but 512-wide routers."""
    from dataclasses import asdict

    from minagi import store
    from minagi.recur import RecurCoder, RecurConfig, _load_dir
    cfg = RecurConfig(vocab_size=265, d_model=8, n_head=1, d_ff=16,
                      n_prelude=1, n_recur=1, n_coda=0, max_steps=1,
                      block=64, use_pool=True, pool_experts=3, pool_d_ff=8,
                      pool_top_k=1, pool_max=16)     # wider than the pool
    src = str(tmp_path / "weights")
    store.save(RecurCoder(cfg), src, step=5, val=None, opt=None,
               cfg=asdict(cfg))
    assert os.path.isdir(os.path.join(src, store.EXPERTS))
    m, meta = _load_dir(src, "cpu")     # build_paged raised on router size
    assert meta["step"] == 5
    assert not hasattr(m.pool, "slots")
    assert m.pool.n_experts() == 3
