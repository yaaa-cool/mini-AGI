"""#19: GradSNR.observe crashed when the set of parameters with a gradient changed."""

import torch

from minagi.optim import GradSNR
from minagi.recur import RecurConfig, RecurCoder


def test_changing_grad_set_counts_missing_as_zero():
    a = torch.nn.Parameter(torch.zeros(5))
    b = torch.nn.Parameter(torch.zeros(3))
    snr = GradSNR()
    a.grad, b.grad = torch.ones(5), torch.ones(3)
    snr.observe([a, b])
    b.grad = None                     # b sat this step out
    snr.observe([a, b])
    assert snr.m.numel() == 8
    assert torch.allclose(snr.m[5:], torch.full((3,), snr.beta))
    a.grad = None
    assert snr.observe([a, b]) is None      # nothing at all: not a reading
    assert snr.n == 2


def test_constant_gradient_reads_one():
    a = torch.nn.Parameter(torch.zeros(4))
    snr = GradSNR()
    for n in range(1, 401):
        a.grad = torch.full((4,), 0.5)
        r = snr.observe([a])
        if n >= 8:                    # from the first reading, not only late
            assert abs(r - 1.0) < 1e-5, (n, r)


def test_noise_reads_below_one_early():
    torch.manual_seed(0)
    a = torch.nn.Parameter(torch.zeros(256))
    snr = GradSNR()
    for _ in range(8):
        a.grad = torch.randn(256)
        r = snr.observe([a])
    assert 0.0 <= r < 1.0


def test_depth_one_step_leaves_halting_head_gradless():
    torch.manual_seed(0)
    cfg = RecurConfig(vocab_size=265, d_model=32, n_head=2, d_ff=64,
                      n_prelude=1, n_recur=1, n_coda=0, max_steps=4,
                      block=16, train_steps_mean=2.0)
    model = RecurCoder(cfg).train()
    params = list(model.parameters())
    x = torch.randint(0, 265, (1, 16))
    snr = GradSNR()
    for depth in (3, 1, 3, 1):
        model.sample_depth = lambda d=depth: d
        model.zero_grad(set_to_none=True)
        _, loss = model(x, x)
        loss.backward()
        if depth == 1:
            assert model.halt.weight.grad is None
        r = snr.observe(params)
    assert snr.m.numel() == sum(p.numel() for p in params)
    assert snr.n == 4 and r is None          # fewer than 8 readings
