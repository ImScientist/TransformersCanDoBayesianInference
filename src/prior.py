"""GP prior for training a PFN.

`get_batch` / `DataLoader` sample datasets from a zero-mean GP with kernel
`bias + variance * x^T x' + rbf_outputscale * RBF(rbf_lengthscale)
 + periodic_outputscale * ExpSineSquared(period, periodic_lengthscale) [* RBF(periodic_decay)]`
(random linear trend + optional smooth deviations + optional (locally) periodic component)
and Gaussian observation noise. Every dataset can get its own hyperparameters drawn from a distribution
(`sample_hyperparameters`).

Two sampling methods (`get_batch(sampling_method=...)`):
  - 'cholesky': build the (batch_size, n, n) covariance (`dot_product_rbf_covariance`) and sample with a
    Cholesky factor; exact, any `num_features`, but O(n^2) memory and O(n^3) time per dataset.
  - 'basis' (`num_features == 1` only): sample every kernel component directly as a random linear combination
    of basis functions (`sample_basis_1d`); O(n) memory and time per dataset, no matrix decomposition:
      linear, bias, noise: exact (`w * x`, constant, iid noise),
      periodic:            exact Fourier series, truncated when the remaining variance is < `basis_tol`,
      RBF:                 Hilbert-space approximation (Solin & Särkkä, 2020), error of order `basis_tol`,
      periodic * RBF:      every harmonic modulated by an independent (Hilbert-space) RBF process.
"""

import math
from statistics import NormalDist

import torch
from linear_operator.utils.cholesky import psd_safe_cholesky
from src.config import Distribution, GPHyperparameters
from src.utils import get_batch_to_dataloader, default_device


def sample_hyperparameters(hyperparameters: GPHyperparameters, batch_size, device=None):
    """Draw one value of every hyperparameter for each of the `batch_size` datasets.

    Every field of `hyperparameters` is either
      - a number (the same value for all datasets),
      - `None` (kept as `None`), or
      - a `Distribution` (`uniform` | `log_uniform` between `low` and `high`, optionally zero-inflated: exactly
        0 with probability `zero_prob`), sampled independently per dataset. `log_uniform` (uniform in `log`,
        requires `low > 0`) suits scale parameters spanning orders of magnitude.

    :return: dict with the field names as keys; numbers and distributions become tensors of shape (batch_size,)
    """
    sampled = {}
    for name in type(hyperparameters).model_fields:
        spec = getattr(hyperparameters, name)
        if spec is None:
            sampled[name] = None
        elif isinstance(spec, Distribution):
            u = torch.rand(batch_size, device=device)
            if spec.dist == 'uniform':
                sampled[name] = spec.low + (spec.high - spec.low) * u
            else:  # log_uniform
                sampled[name] = torch.exp(math.log(spec.low) + (math.log(spec.high) - math.log(spec.low)) * u)
            if spec.zero_prob > 0:
                is_zero = torch.rand(batch_size, device=device) < spec.zero_prob
                sampled[name] = sampled[name].masked_fill(is_zero, 0.)
        else:
            sampled[name] = torch.full((batch_size,), float(spec), device=device)
    return sampled


def _active(scale):
    """True if a kernel component with this (batch of) scale(s) contributes to any dataset."""
    return scale is not None and bool((scale > 0).any())


def dot_product_rbf_covariance(x, variance, rbf_outputscale=None, rbf_lengthscale=None, bias=None,
                               periodic_outputscale=None, period=None, periodic_lengthscale=None,
                               periodic_decay=None):
    """Prior covariance (without noise) with a different set of hyperparameters per dataset.

    `K_b = bias_b + variance_b * x_b x_b^T
           + rbf_outputscale_b * exp(-||x_b - x_b'||^2 / (2 * rbf_lengthscale_b^2))
           + periodic_outputscale_b * P_b(x, x') * D_b(x, x')`,
    i.e. a random linear trend `c + w^T x` with `c ~ N(0, bias)`, `w ~ N(0, variance * I)` plus independent
    smooth RBF deviations plus an independent (locally) periodic component with
      - `P_b = exp(-2 * sum_i sin^2(pi * |x_i - x'_i| / period_b) / periodic_lengthscale_b^2)`
        (Exp-Sine-Squared, summed over features like gpytorch's `PeriodicKernel`, which keeps it PSD for
        `num_features > 1`; sklearn's Euclidean-distance version is not PSD in general; note gpytorch does
        not square its lengthscale, i.e. gpytorch `lengthscale = periodic_lengthscale^2`) and
      - `D_b = exp(-||x - x'||^2 / (2 * periodic_decay_b^2))`, an optional RBF factor that lets the periodic
        pattern change over distances of order `periodic_decay` (`None` = exactly periodic).

    :param x: inputs, shape (batch_size, n, num_features)
    :param variance, rbf_*, bias, periodic_*, period: tensors of shape (batch_size,);
        `rbf_outputscale`, `bias`, `periodic_outputscale` may be `None` (= 0), `periodic_decay` may be `None`
    :return: covariance matrices, shape (batch_size, n, n)
    """

    def b(v):
        return v[:, None, None]

    active = _active

    # -> (batch_size, n, n)
    covar = b(variance) * (x @ x.transpose(-1, -2))

    if bias is not None:
        # -> (batch_size, n, n)
        covar = covar + b(bias)

    if not (active(rbf_outputscale) or active(periodic_outputscale)):
        return covar

    diff = x[:, :, None, :] - x[:, None, :, :]  # -> (batch_size, n, n, num_features)
    sq_dist = diff.square().sum(-1)
    if active(rbf_outputscale):
        if rbf_lengthscale is None:
            raise ValueError('`rbf_lengthscale` is required when `rbf_outputscale > 0`.')
        covar = covar + b(rbf_outputscale) * torch.exp(-.5 * sq_dist / b(rbf_lengthscale).square())

    if active(periodic_outputscale):
        if period is None or periodic_lengthscale is None:
            raise ValueError('`period` and `periodic_lengthscale` are required when `periodic_outputscale > 0`.')
        # -> (batch_size, n, n)
        sin_sq = torch.sin(math.pi * diff.abs() / period[:, None, None, None]).square().sum(-1)
        periodic = torch.exp(-2. * sin_sq / b(periodic_lengthscale).square())

        if periodic_decay is not None:
            periodic = periodic * torch.exp(-.5 * sq_dist / b(periodic_decay).square())

        covar = covar + b(periodic_outputscale) * periodic

    return covar


def rbf_basis_1d(x, lengthscale, tol=1e-4):
    """Hilbert-space approximation (Solin & Särkkä, 2020) of the unit-variance RBF kernel in 1D.

    Returns features `phi` with `phi @ phi^T ≈ exp(-(x - x')^2 / (2 * lengthscale^2))`, i.e. `phi @ N(0, I)` is
    a sample of the RBF process. The basis are the Dirichlet eigenfunctions `sin(omega_j (x - c + L)) / sqrt(L)`,
    `omega_j = pi j / (2 L)`, of an interval `[c - L, c + L]` around the data, weighted by the square root of the
    RBF spectral density `S(omega) = sqrt(2 pi) l exp(-l^2 omega^2 / 2)`. Both error sources are of order `tol`:
      - boundary (f is pinned to 0 at c +- L): `L = half data range + l * sqrt(ln(1/tol) / 2)`, so the
        reflected (method of images) correlation `exp(-2 (L - half range)^2 / l^2)` is `tol`,
      - truncation after M basis functions: `omega_M >= z / l` with `P(|N(0, 1)| > z) = tol`.
    M is shared by the batch (the largest one needed); short lengthscales need more (~ 2.5 / l for tol=1e-4).

    :param x: (batch_size, n); :param lengthscale: (batch_size,)
    :return: (batch_size, n, M)
    """
    lo, hi = x.amin(1), x.amax(1)
    center, half_range = (lo + hi) / 2, (hi - lo) / 2
    z = NormalDist().inv_cdf(1 - tol / 2)
    L = half_range + lengthscale * math.sqrt(math.log(1 / tol) / 2)
    num_basis = max(1, math.ceil((2 * L * z / (math.pi * lengthscale)).max().item()))
    # -> (batch_size, M)
    omega = math.pi * torch.arange(1, num_basis + 1, device=x.device, dtype=x.dtype) / (2 * L[:, None])
    spectral_density = math.sqrt(2 * math.pi) * lengthscale[:, None] * torch.exp(
        -.5 * (lengthscale[:, None] * omega).square())
    # -> (batch_size, n, M)
    phi = torch.sin(omega[:, None, :] * (x - center[:, None] + L[:, None])[:, :, None]) / L.sqrt()[:, None, None]
    return phi * spectral_density.sqrt()[:, None, :]


def periodic_fourier_coefficients(lengthscale, tol=1e-4):
    """Fourier coefficients of the unit-variance Exp-Sine-Squared kernel.

    `exp(-2 sin^2(pi tau / period) / l^2) = sum_k c_k cos(2 pi k tau / period)` with
    `c_0 = exp(-a) I_0(a)`, `c_k = 2 exp(-a) I_k(a)` (`a = 1 / l^2`, `I_k` modified Bessel functions), so
    `sum_k c_k = 1`. The coefficients are computed with an FFT of the kernel over one period and truncated after the
    first K where the remaining variance `1 - sum_{k <= K} c_k` is <= `tol` for every dataset
    (K ~ 4 sqrt(a); e.g. 8 harmonics for l = 0.5, tol=1e-4).

    :param lengthscale: (batch_size,) `periodic_lengthscale`
    :return: (batch_size, K + 1), harmonics k = 0..K
    """
    a = lengthscale.pow(-2)
    grid_size = 2 ** math.ceil(math.log2(16 * math.sqrt(a.max().item()) + 64))  # aliasing far below tol
    t = 2 * math.pi * torch.arange(grid_size, device=a.device, dtype=a.dtype) / grid_size
    c = torch.fft.rfft(torch.exp(a[:, None] * (torch.cos(t) - 1)), dim=-1).real / grid_size
    c = c[:, :grid_size // 2].clamp_min(0)  # drop the Nyquist term; clamp round-off
    c[:, 1:] *= 2  # cos(k t) collects the +k and -k terms
    remaining = 1 - c.cumsum(-1)  # variance missing when stopping after harmonic k
    num_harmonics = min(int((remaining.amax(0) > tol).sum()) + 1, c.shape[-1])
    return c[:, :num_harmonics]


def sample_basis_1d(x, variance, rbf_outputscale=None, rbf_lengthscale=None, bias=None,
                    periodic_outputscale=None, period=None, periodic_lengthscale=None,
                    periodic_decay=None, tol=1e-4):
    """Sample f(x) (without noise) of the `dot_product_rbf_covariance` prior for `num_features == 1`, without
    building a covariance matrix: every kernel component is an independent random function

      - linear:   `sqrt(variance) w x`, `w ~ N(0, 1)` (exact)
      - bias:     `sqrt(bias) z`, `z ~ N(0, 1)` (exact)
      - RBF:      `sqrt(rbf_outputscale) phi @ beta`, `beta ~ N(0, I)`, `phi = rbf_basis_1d(x, rbf_lengthscale)`
      - periodic: `sqrt(periodic_outputscale) sum_k sqrt(c_k) (alpha_k cos(k theta) + beta_k sin(k theta))`,
        `theta = 2 pi x / period`, `c = periodic_fourier_coefficients(periodic_lengthscale)` (exact up to truncation);
        with `periodic_decay` the random amplitudes `alpha_k, beta_k` become independent RBF processes
        `h_k(x)` with lengthscale `periodic_decay` (covariance `g(x - x') * sum_k c_k cos(k (theta - theta'))`, i.e.
        exactly the locally periodic kernel)

    Arguments as in `dot_product_rbf_covariance`, with x of shape (batch_size, n).
    :param tol: accuracy of the RBF approximations and of the Fourier truncation (variance units of a
        unit-scale component)
    :return: (batch_size, n)
    """
    batch_size, n = x.shape
    device, dtype = x.device, x.dtype

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype)

    f = torch.zeros_like(x)
    if variance is not None:
        f = f + (variance.sqrt() * randn(batch_size))[:, None] * x
    if bias is not None:
        f = f + (bias.sqrt() * randn(batch_size))[:, None]

    if _active(rbf_outputscale):
        if rbf_lengthscale is None:
            raise ValueError('`rbf_lengthscale` is required when `rbf_outputscale > 0`.')
        phi = rbf_basis_1d(x, rbf_lengthscale, tol)
        f = f + rbf_outputscale.sqrt()[:, None] * (phi @ randn(batch_size, phi.shape[-1], 1)).squeeze(-1)

    if _active(periodic_outputscale):
        if period is None or periodic_lengthscale is None:
            raise ValueError('`period` and `periodic_lengthscale` are required when `periodic_outputscale > 0`.')
        sqrt_c = periodic_fourier_coefficients(periodic_lengthscale, tol).sqrt()  # (batch_size, K + 1)
        k = torch.arange(sqrt_c.shape[-1], device=device, dtype=dtype)
        k_theta = (2 * math.pi * x / period[:, None])[:, :, None] * k  # (batch_size, n, K + 1)
        if periodic_decay is None:
            amplitudes = randn(batch_size, 1, 2 * len(k))  # (alpha_k, beta_k), the same for all x
        else:
            phi = rbf_basis_1d(x, periodic_decay, tol)
            amplitudes = phi @ randn(batch_size, phi.shape[-1], 2 * len(k))  # (batch_size, n, 2 (K + 1))
        alpha, beta = amplitudes.split(len(k), dim=-1)
        periodic = (sqrt_c[:, None, :] * (alpha * torch.cos(k_theta) + beta * torch.sin(k_theta))).sum(-1)
        f = f + periodic_outputscale.sqrt()[:, None] * periodic

    return f


@torch.no_grad()
# WARNING: `default_device` is 'mps' on Apple Silicon. MPS has no float64 support, so GP prior sampling with
# sampling_method='cholesky' can be numerically unstable for very small noise (e.g. noise=1e-4). If you see
# NotPSD/jitter warnings or errors, pass device='cpu' here.
def get_batch(
        batch_size,
        seq_len: int,
        num_features: int,
        device=default_device,
        hyperparameters: GPHyperparameters | dict | None = None,
        equidistant_x: bool = False,
        sampling_method: str = 'auto',
        basis_tol: float = 1e-4
):
    """Sample `batch_size` datasets from the GP prior (zero mean, kernel of `dot_product_rbf_covariance`).

    Every dataset gets its own hyperparameters (`noise`, `variance`, `rbf_outputscale`, `rbf_lengthscale`, `bias`,
    `periodic_outputscale`, `period`, `periodic_lengthscale`, `periodic_decay`) drawn by
    `sample_hyperparameters`, so a hyperparameter given as a distribution varies from dataset to dataset
    (fixed numbers reproduce a single GP prior).

    :param hyperparameters: `GPHyperparameters`, or a dict validated into one (missing keys take the
        `GPHyperparameters` defaults, unknown keys and invalid values raise); `None` = all defaults
    :param sampling_method: 'cholesky' (exact, any `num_features`, O(seq_len^3)), 'basis' (`sample_basis_1d`,
        `num_features == 1` only, O(seq_len)) or 'auto' ('basis' if `num_features == 1`, else 'cholesky')
    :param basis_tol: accuracy of the 'basis' method (see `sample_basis_1d`)
    :return: x: (seq_len, batch_size, num_features), y and target_y (identical): (seq_len, batch_size)
    """
    if not isinstance(hyperparameters, GPHyperparameters):
        hyperparameters = GPHyperparameters.model_validate(hyperparameters or {})
    if sampling_method == 'auto':
        sampling_method = 'basis' if num_features == 1 else 'cholesky'
    if sampling_method not in ('basis', 'cholesky'):
        raise ValueError(f"sampling_method must be 'auto', 'basis' or 'cholesky', got {sampling_method!r}")
    if sampling_method == 'basis' and num_features != 1:
        raise ValueError("sampling_method='basis' requires num_features == 1")

    # -> (batch_size, seq_len, num_feat)
    if equidistant_x:
        assert num_features == 1
        x = torch.linspace(0, 1, seq_len, device=device)[:, None].repeat(batch_size, 1, 1)
    else:
        x = torch.rand(batch_size, seq_len, num_features, device=device)

    hp = sample_hyperparameters(hyperparameters, batch_size, device=device)
    kernel_hp = (hp['variance'], hp['rbf_outputscale'], hp['rbf_lengthscale'], hp['bias'],
                 hp['periodic_outputscale'], hp['period'], hp['periodic_lengthscale'], hp['periodic_decay'])

    if sampling_method == 'basis':
        f = sample_basis_1d(x[..., 0], *kernel_hp, tol=basis_tol)
        sample = (f + hp['noise'].sqrt()[:, None] * torch.randn_like(f)).transpose(0, 1)
    else:
        covar = dot_product_rbf_covariance(x, *kernel_hp)
        # Observation noise on the diagonal -> covariance of y.
        covar = covar + hp['noise'][:, None, None] * torch.eye(seq_len, device=device)

        chol = psd_safe_cholesky(covar)  # adds jitter if needed
        sample = (chol @ torch.randn(batch_size, seq_len, 1, device=device)).squeeze(-1).transpose(0, 1)

    # -> (seq_len, batch_size, num_feat), (seq_len, batch_size), (seq_len, batch_size)
    return x.transpose(0, 1), sample, sample  # x.shape = (T,B,H)


DataLoader = get_batch_to_dataloader(get_batch)
