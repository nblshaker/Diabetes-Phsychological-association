"""
Privacy accountant for DP-SGD (numpy/scipy only - no PyTorch/Opacus needed).

DP-SGD at each step: Poisson-sample a batch with rate q, clip every example's gradient to norm C, add Gaussian
noise N(0, sigma^2 C^2). Over T steps this is the "sampled Gaussian mechanism" composed T times.

Method: Renyi-DP (Mironov, Talwar, Zhang 2019) at integer orders alpha, composed by multiplying by T, then
converted to (epsilon, delta) with the tighter conversion of Balle et al. 2020 (the same one Opacus uses):

    A_alpha   = sum_{k=0..alpha} C(alpha,k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2))
    RDP(alpha)= T * log(A_alpha) / (alpha - 1)
    epsilon   = min_alpha  RDP(alpha) - (log(delta) + log(alpha)) / (alpha - 1) + log((alpha-1)/alpha)

Integer orders only, so the bound can be marginally looser than Opacus' fractional-order grid (checked in
tests_dp.py: agreement is within a few percent).
"""
import functools
import numpy as np
from scipy.special import gammaln, logsumexp

ORDERS = np.arange(2, 257)


def _log_binom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rdp_one_step(q, sigma, alpha):
    """log A_alpha / (alpha - 1) for one sampled-Gaussian step at integer order alpha."""
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    k = np.arange(0, alpha + 1)
    log_terms = (_log_binom(alpha, k) + (alpha - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma ** 2))
    return float(logsumexp(log_terms)) / (alpha - 1)


def epsilon(q, sigma, steps, delta, orders=ORDERS):
    eps = []
    for a in orders:
        rdp = steps * rdp_one_step(q, sigma, int(a))
        eps.append(rdp - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
    return float(max(min(eps), 0.0))


@functools.lru_cache(maxsize=None)
def noise_for_target(target_eps, delta, q, steps, tol=1e-3):
    """Smallest sigma such that `steps` DP-SGD steps at sampling rate q satisfy (target_eps, delta)-DP (bisection)."""
    lo, hi = 0.2, 2.0
    while epsilon(q, hi, steps, delta) > target_eps:
        hi *= 2
        if hi > 1e4:
            raise ValueError("target epsilon unreachable")
    for _ in range(60):
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi
