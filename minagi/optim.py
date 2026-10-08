"""
A meter for how much of a gradient is signal.
"""

import torch


class GradSNR:
    """
    How much of the gradient is signal, measured every step.

    The held-out signal this project steers the learning rate by moves 0.0014
    per evaluation against 0.018 of noise, so it takes over a hundred readings
    to say anything - twenty hours at a ten-minute checkpoint. This is the
    same question asked of a quantity that is available on every step.

    Keep an average of the gradient and an average of its squared norm. If
    successive gradients agree, the average keeps its length and the ratio
    ||mean||^2 / mean(||g||^2) approaches one. If they are independent noise
    the average shrinks toward zero and so does the ratio. It is the gradient
    noise scale of McCandlish et al. 2018, in the cheapest form that answers
    the question: two scalars, no extra tensors.

    Reported, not acted on. A signal is watched for a while before anything is
    allowed to steer on it.
    """

    def __init__(self, beta=0.98):
        self.beta = beta
        self.m = None
        self.sq = 0.0
        self.n = 0
        self._sums = []     # squared norms still on the device, oldest first

    @torch.no_grad()
    def observe(self, params, report=True):
        # `report=False` returns nothing and leaves the squared norm on the
        # device until the ratio is asked for, so the step does not wait.
        # A parameter with no gradient contributed a zero gradient this step,
        # so it is counted as zeros rather than left out. Leaving it out made
        # the flattened vector change length whenever a parameter sat a step
        # out - which recurrence depth sampling makes routine: a step sampled
        # at depth 1 forces the halt, and the halting head receives nothing -
        # and the running mean then no longer matched it.
        if all(p.grad is None for p in params):
            return None
        flat = torch.cat([(p.grad if p.grad is not None
                           else torch.zeros_like(p)).detach().float().reshape(-1)
                          for p in params])
        self.m = flat.clone() if self.m is None else \
            self.m.mul_(self.beta).add_(flat, alpha=1 - self.beta)
        self._sums.append((flat * flat).sum())
        self.n += 1
        if len(self._sums) >= 64:
            self._fold()
        return self.ratio() if report else None

    def _fold(self):
        """Read the pending squared norms back into sq, in the order taken."""
        if not self._sums:
            return
        first = self.n - len(self._sums) == 0
        for s in torch.stack(self._sums).tolist():
            self.sq = s if first else self.beta * self.sq + (1 - self.beta) * s
            first = False
        self._sums.clear()

    def ratio(self):
        """0 = pure noise, 1 = every step pointing the same way."""
        self._fold()
        if self.m is None or self.sq <= 0 or self.n < 8:
            return None
        # No bias correction: both averages start AT the first reading, not at
        # zero, so neither is biased toward zero. Dividing by 1 - beta^n here
        # inflated the ratio by that factor - 6.7x at the eighth reading, and
        # a constant gradient read above one.
        return float(self.m.pow(2).sum() / self.sq)
