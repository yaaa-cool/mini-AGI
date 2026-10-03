"""#18: measuring with a chunk longer than the context ran past the rotary tables."""

import numpy as np
import torch

from minagi.recur import RecurConfig, RecurCoder
from minagi.stream import FileReader


def tiny(block):
    torch.manual_seed(0)
    cfg = RecurConfig(vocab_size=265, d_model=32, n_head=2, d_ff=64,
                      n_prelude=1, n_recur=1, n_coda=0, max_steps=2,
                      block=block)
    return RecurCoder(cfg).eval()


def test_measure_chunk_longer_than_context():
    context = 16
    model = tiny(block=context)
    data = (np.arange(200) % 251).astype(np.uint8)    # longer than the chunk
    r = FileReader(model, data, "f", chunk=64, context=context, device="cpu")
    losses = []
    with torch.no_grad():
        while not r.done():
            loss = r.step(learn=False)
            if loss is None:
                break
            losses.append(float(loss))
    assert losses and all(np.isfinite(losses))
    assert r.pos >= len(data) - 2              # the whole file was scored


def test_learning_step_unchanged():
    context = 16
    model = tiny(block=context).train()
    data = (np.arange(100) % 251).astype(np.uint8)
    r = FileReader(model, data, "f", chunk=64, context=context, device="cpu")
    loss = r.step(learn=True)
    assert torch.isfinite(loss) and r.pos == 64
