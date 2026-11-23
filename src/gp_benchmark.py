"""Benchmark: compare a trained PFN with the exact GP posterior predictive.

A PFN trained on a GP prior should approximate the posterior predictive distribution of that GP. For fixed
hyperparameters the exact posterior predictive is known in closed form, so the PFN can be checked directly.

The benchmark file (`create`) holds several subsets, each with fixed GP hyperparameters (`SUBSETS`) and one
task: `interpolation` (x uniform in [0, 1], random order) or `forecast` (x sorted, the context consists of the
smallest x). Every subset has `num_datasets` sequences of `seq_len` points sampled exactly (Cholesky, float64, CPU,
fixed seed); it stores `x`, `y` and, in its metadata, the hyperparameters and creation settings. The data are
stored rather than regenerated from a seed because random numbers differ between devices and torch versions and
because later changes to `src/prior.py` must not change the benchmark. The exact posterior is computed at evaluation
time (float64, deterministic, fast for these sizes) from `x`, `y` and the hyperparameters.

Evaluation (`evaluate`): for every context size n in `context_sizes`, the first n points are the context and the
remaining `seq_len - n` points the queries (as in training with `single_eval_pos = n`). Metrics per query point,
averaged over query points and datasets:
  - `kl`:       KL(exact GP || PFN) of the predictive distributions of y (0 = identical; the main metric)
  - `nll_pfn`:  Gaussian NLL of the true y under the PFN (the training loss)
  - `nll_gp`:   the same under the exact GP (the Bayes-optimal value); `nll_gap = nll_pfn - nll_gp`
  - `mean_z`:   RMS of (mean_pfn - mean_gp) / std_gp (error of the mean in units of the posterior std)
  - `log_var_ratio`: mean of log(var_pfn / var_gp) (> 0: the PFN is too uncertain, < 0: overconfident)

    python -m src.gp_benchmark create --output benchmarks/gp_benchmark_v1.npz
    python -m src.gp_benchmark evaluate --checkpoint outputs/my_run_1 [--output outputs/my_run_1/gp_benchmark.yaml]

During training, `training.gp_benchmark=FILE` (see `src/train.py`) evaluates the benchmark every
`training.gp_benchmark_every` epochs and logs the results (`summarize`, per subset, per context size) to TensorBoard.

The hyperparameters of `SUBSETS` lie inside the ranges of `configs/gp_hyperprior.yaml`; a model trained on a
narrower prior can only be expected to do well on the subsets its prior covers.

The exact GP knows the true hyperparameters of each subset, so it is a best-case reference (a lower bound of the
expected NLL), not the exact target: a PFN trained with a hyperprior approximates the posterior predictive averaged
over the hyperprior, which is wider, especially for small contexts (e.g. at n = 0 it is the hyperprior's prior
predictive). `kl` and `nll_gap` are therefore > 0 even for a perfect PFN and should be compared between models; they
go to 0 only for a model trained on exactly the subset's fixed hyperparameters.
"""
import argparse
import datetime
import json
import math
import os

import gpytorch
import numpy as np
import torch
import yaml

from src.config import GPHyperparameters
from src.prior import dot_product_rbf_covariance
from src.utils import default_device

VERSION = 1
SEQ_LEN = 64
NUM_DATASETS = 256
CONTEXT_SIZES = (0, 1, 2, 4, 8, 16, 32, 48)
SEED = 0
TASKS = ('interpolation', 'forecast')

# Fixed hyperparameters per subset (unspecified fields take the `GPHyperparameters` defaults); every subset is
# generated once per task in `TASKS`. Changing these requires a new VERSION (and benchmark file).
_NO_TREND = {'variance': 0., 'bias': 0.}
SUBSETS = {
    'rbf_short': {**_NO_TREND, 'rbf_outputscale': 1., 'rbf_lengthscale': .05, 'noise': .01},
    'rbf_long': {**_NO_TREND, 'rbf_outputscale': 1., 'rbf_lengthscale': .5, 'noise': .01},
    'rbf_noisy': {**_NO_TREND, 'rbf_outputscale': 1., 'rbf_lengthscale': .15, 'noise': .1},
    'rbf_low_noise': {**_NO_TREND, 'rbf_outputscale': 1., 'rbf_lengthscale': .15, 'noise': 1e-4},
    'linear': {'variance': 1., 'bias': 1., 'rbf_outputscale': 0., 'noise': .01},
    'linear_rbf': {'variance': 1., 'bias': 1., 'rbf_outputscale': .25, 'rbf_lengthscale': .15, 'noise': .01},
    'periodic': {**_NO_TREND, 'rbf_outputscale': 0., 'periodic_outputscale': 1., 'period': .2,
                 'periodic_lengthscale': 1., 'noise': .01},
    'linear_periodic': {'variance': 1., 'bias': 1., 'rbf_outputscale': 0., 'periodic_outputscale': .5,
                        'period': .2, 'periodic_lengthscale': 1., 'noise': .01},
}

_KERNEL_ARGS = ('variance', 'rbf_outputscale', 'rbf_lengthscale', 'bias', 'periodic_outputscale', 'period',
                'periodic_lengthscale', 'periodic_decay')


def prior_covariance(x, hyperparameters):
    """Covariance of y (kernel + noise) in float64.

    :param x: (num_datasets, n), :param hyperparameters: dict of fixed values (as stored in the benchmark)
    :return: (num_datasets, n, n)
    """
    x = x.to(torch.float64)
    batch_size, n = x.shape

    def full(name):
        v = hyperparameters[name]
        return None if v is None else torch.full((batch_size,), float(v), dtype=torch.float64)

    covar = dot_product_rbf_covariance(x[..., None], *(full(name) for name in _KERNEL_ARGS))
    return covar + hyperparameters['noise'] * torch.eye(n, dtype=torch.float64)


def gp_posterior(covar, y, n):
    """Exact posterior predictive of y at positions n.. given y at positions ..n (float64).

    :param covar: (num_datasets, seq_len, seq_len) covariance of y; :param y: (num_datasets, seq_len)
    :return: mean, var, each (num_datasets, seq_len - n)
    """
    y = y.to(torch.float64)
    k_qq = covar[:, n:, n:].diagonal(dim1=-2, dim2=-1)
    if n == 0:
        return torch.zeros_like(k_qq), k_qq
    chol = torch.linalg.cholesky(covar[:, :n, :n])
    k_cq = covar[:, :n, n:]
    a = torch.cholesky_solve(y[:, :n, None], chol)  # K_cc^-1 y_c
    v = torch.linalg.solve_triangular(chol, k_cq, upper=False)  # L^-1 K_cq
    mean = (k_cq.transpose(-1, -2) @ a).squeeze(-1)
    var = k_qq - v.square().sum(-2)
    return mean, var


def create_benchmark(path, seed=SEED, num_datasets=NUM_DATASETS, seq_len=SEQ_LEN):
    """Sample all subsets exactly (Cholesky, float64, CPU) and save them as `.npz` (x, y as float32)."""
    generator = torch.Generator().manual_seed(seed)
    arrays, subsets = {}, {}
    for name, overrides in SUBSETS.items():
        hyperparameters = GPHyperparameters(**overrides).model_dump()
        for task in TASKS:
            x = torch.rand(num_datasets, seq_len, dtype=torch.float64, generator=generator)
            if task == 'forecast':
                x = x.sort(dim=1).values
            # Round x to float32 first, so that the stored x is exactly the x the sample belongs to.
            x = x.to(torch.float32).to(torch.float64)
            chol = torch.linalg.cholesky(prior_covariance(x, hyperparameters))
            eps = torch.randn(num_datasets, seq_len, 1, dtype=torch.float64, generator=generator)
            y = (chol @ eps).squeeze(-1)
            key = f'{name}/{task}'
            arrays[f'{key}/x'] = x.to(torch.float32).numpy()
            arrays[f'{key}/y'] = y.to(torch.float32).numpy()
            subsets[key] = {'hyperparameters': hyperparameters, 'task': task}
    metadata = {
        'version': VERSION, 'seed': seed, 'num_datasets': num_datasets, 'seq_len': seq_len,
        'context_sizes': list(CONTEXT_SIZES), 'subsets': subsets,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'torch': str(torch.__version__), 'gpytorch': str(gpytorch.__version__),
    }
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)
    return metadata


def load_benchmark(path):
    """:return: (metadata dict, {subset: (x, y)} with float32 tensors of shape (num_datasets, seq_len))"""
    with np.load(path) as data:
        metadata = json.loads(str(data['metadata']))
        subsets = {key: (torch.from_numpy(data[f'{key}/x']), torch.from_numpy(data[f'{key}/y']))
                   for key in metadata['subsets']}
    return metadata, subsets


def gaussian_metrics(mean_pfn, var_pfn, mean_gp, var_gp, y):
    """Per-point metrics (see module docstring), averaged over all entries; all inputs float64, same shape."""
    log_2pi = math.log(2 * math.pi)
    nll_pfn = .5 * (torch.log(var_pfn) + (y - mean_pfn).square() / var_pfn + log_2pi)
    nll_gp = .5 * (torch.log(var_gp) + (y - mean_gp).square() / var_gp + log_2pi)
    kl = .5 * (torch.log(var_pfn / var_gp) + (var_gp + (mean_gp - mean_pfn).square()) / var_pfn - 1)
    return {
        'kl': kl.mean().item(),
        'nll_pfn': nll_pfn.mean().item(),
        'nll_gp': nll_gp.mean().item(),
        'nll_gap': (nll_pfn - nll_gp).mean().item(),
        'mean_z': ((mean_pfn - mean_gp).square() / var_gp).mean().sqrt().item(),
        'log_var_ratio': torch.log(var_pfn / var_gp).mean().item(),
    }


@torch.no_grad()
def evaluate(model, benchmark_path, context_sizes=None):
    """Compare `model` with the exact GP on every subset and context size.

    :param model: PFN in eval mode (e.g. from `src.train.load_checkpoint`)
    :return: {subset: {'mean': metrics averaged over context sizes, 'by_context_size': {n: metrics}}}
    """
    metadata, subsets = load_benchmark(benchmark_path)
    context_sizes = list(context_sizes or metadata['context_sizes'])
    device = next(model.parameters()).device
    results = {}
    for key, (x, y) in subsets.items():
        covar = prior_covariance(x, metadata['subsets'][key]['hyperparameters'])
        x_model, y_model = x.T[..., None].to(device), y.T.to(device)  # (seq_len, num_datasets[, 1])
        by_size = {}
        for n in context_sizes:
            mean_gp, var_gp = gp_posterior(covar, y, n)
            mean_pfn, var_pfn = model((x_model, y_model), single_eval_pos=n)
            by_size[n] = gaussian_metrics(mean_pfn.T.cpu().double(), var_pfn.T.cpu().double(),
                                          mean_gp, var_gp, y[:, n:].double())
        results[key] = {'mean': {m: float(np.mean([r[m] for r in by_size.values()])) for m in by_size[n]},
                        'by_context_size': by_size}
    return results


def summarize(results):
    """Metrics of `evaluate` averaged over all subsets (and context sizes): {metric: value}."""
    metric_names = next(iter(results.values()))['mean']
    return {m: float(np.mean([r['mean'][m] for r in results.values()])) for m in metric_names}


def format_results(results):
    """Text tables: metrics averaged over context sizes per subset, and KL per context size."""
    metric_names = list(next(iter(results.values()))['mean'])
    sizes = list(next(iter(results.values()))['by_context_size'])
    width = max(len(k) for k in results) + 2
    lines = ['Averaged over context sizes:',
             f"{'subset':<{width}}" + ''.join(f'{m:>14}' for m in metric_names)]
    lines += [f'{key:<{width}}' + ''.join(f"{r['mean'][m]:>14.4f}" for m in metric_names) for key, r in results.items()]
    summary = summarize(results)
    lines += [f"{'all subsets':<{width}}" + ''.join(f'{summary[m]:>14.4f}' for m in metric_names)]
    lines += ['', 'KL(exact GP || PFN) by context size:',
              f"{'subset':<{width}}" + ''.join(f'{n:>9}' for n in sizes)]
    lines += [f'{key:<{width}}' + ''.join(f"{r['by_context_size'][n]['kl']:>9.4f}" for n in sizes)
              for key, r in results.items()]
    return '\n'.join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Compare a PFN with the exact GP posterior predictive.')
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create', help='Sample the benchmark datasets and save them (.npz).')
    create.add_argument('--output', default=f'benchmarks/gp_benchmark_v{VERSION}.npz')
    create.add_argument('--seed', type=int, default=SEED)
    evaluate_ = sub.add_parser('evaluate', help='Evaluate a trained run on the benchmark.')
    evaluate_.add_argument('--checkpoint', required=True, help='Run folder or .pt file (see src.train.load_checkpoint).')
    evaluate_.add_argument('--benchmark', default=f'benchmarks/gp_benchmark_v{VERSION}.npz')
    evaluate_.add_argument('--context_sizes', type=int, nargs='+', help='Default: those of the benchmark file.')
    evaluate_.add_argument('--device', default=default_device)
    evaluate_.add_argument('--output', help='Optional YAML file for the results.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == 'create':
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        metadata = create_benchmark(args.output, seed=args.seed)
        print(f"Saved {len(metadata['subsets'])} subsets x {metadata['num_datasets']} datasets "
              f"(seq_len {metadata['seq_len']}) to {args.output}")
        return

    from src.train import load_checkpoint
    model, config, _ = load_checkpoint(args.checkpoint, device=args.device)
    seq_len = load_benchmark(args.benchmark)[0]['seq_len']
    if config.training.bptt < seq_len:
        print(f'Warning: the model was trained on sequences of length {config.training.bptt} < {seq_len} '
              f'(benchmark seq_len); longer sequences are outside its training distribution.')
    results = evaluate(model, args.benchmark, args.context_sizes)
    print(format_results(results))
    if args.output:
        with open(args.output, 'w') as f:
            yaml.safe_dump({'checkpoint': args.checkpoint, 'benchmark': args.benchmark, 'results': results}, f,
                           sort_keys=False)
        print(f'Saved results to {args.output}')


if __name__ == '__main__':
    main()
