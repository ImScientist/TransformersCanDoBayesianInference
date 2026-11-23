"""Evaluate a trained PFN on the M4 competition data (one frequency, e.g. the 23,000 yearly series).

Data: `datasets/m4/{Train,Test}/<Frequency>-{train,test}.csv` from github.com/Mcompetitions/M4-methods (Dataset/);
download with `python -m src.download_m4 [--frequencies Yearly]`.

Per series:
  - context: the last `min(len, bptt - horizon)` observations (the model has seen at most `bptt` points),
  - x: positions of context + horizon rescaled to [0, 1] (the training range of the prior),
  - y: standardised with mean and std of the context (std 0 -> 1); predictions are transformed back,
  - forecast: Gaussian mean and the central 95% interval for the `horizon` test points.

Metrics as in the competition: sMAPE, MASE (scale: mean |y_t - y_{t-m}| of the whole training series),
OWA relative to Naive2 and MSIS of the 95% interval. For yearly/weekly/daily data (m = 1) Naive2 equals
Naive1 (last value); for the others the official Naive2 is seasonally adjusted, which is not implemented,
so OWA is reported relative to the seasonal naive there. Also reported: Gaussian NLL in the standardised
space and the coverage of the 95% interval.

Run from the repository root:
    python -m src.evaluate_m4 --checkpoint outputs/my_run --frequency Yearly
"""
import argparse
import csv
import os

import numpy as np
import torch

from src.inference import predict
from src.train import load_checkpoint

# frequency -> (horizon, seasonal period m)
M4_FREQUENCIES = {'Yearly': (6, 1), 'Quarterly': (8, 4), 'Monthly': (18, 12),
                  'Weekly': (13, 1), 'Daily': (14, 1), 'Hourly': (48, 24)}
Z_95 = 1.959964


def load_m4(data_dir, frequency):
    """:return: ids, train series, test series (lists of 1D float arrays)"""

    def read(path):
        with open(path, newline='') as f:
            rows = list(csv.reader(f))[1:]
        return [r[0] for r in rows], [np.array([float(v) for v in r[1:] if v != ''], dtype=np.float64)
                                      for r in rows]

    ids, train = read(os.path.join(data_dir, 'Train', f'{frequency}-train.csv'))
    test_ids, test = read(os.path.join(data_dir, 'Test', f'{frequency}-test.csv'))
    assert ids == test_ids, 'train and test files list different series.'
    return ids, train, test


def pfn_forecast(model, train, horizon, max_context, batch_size=256):
    """Forecast `horizon` steps of every series with the PFN.

    :return: mean, lower, upper (95%), each (num_series, horizon), and the standardised (mean, var, scale) needed
        for the NLL: list of `(mean, var)` tensors per series and arrays `loc`, `scale` (num_series,)
    """
    contexts, queries, loc, scale = [], [], [], []
    for series in train:
        y = series[-max_context:]
        n = len(y)
        x = torch.arange(n + horizon, dtype=torch.float32)[:, None] / (n + horizon - 1)
        mu, sd = y.mean(), y.std()
        sd = sd if sd > 0 else 1.
        contexts.append((x[:n], torch.tensor((y - mu) / sd, dtype=torch.float32)))
        queries.append(x[n:])
        loc.append(mu)
        scale.append(sd)
    predictions = predict(model, contexts, queries, batch_size=batch_size)

    loc, scale = np.array(loc)[:, None], np.array(scale)[:, None]
    mean_std = torch.stack([m for m, _ in predictions]).double().numpy()
    sd_std = torch.stack([v for _, v in predictions]).double().sqrt().numpy()
    mean = loc + scale * mean_std
    lower = loc + scale * (mean_std - Z_95 * sd_std)
    upper = loc + scale * (mean_std + Z_95 * sd_std)
    return mean, lower, upper, predictions, loc, scale


def naive_forecasts(train, horizon, m):
    """Naive (last value), seasonal naive (last season repeated) and drift (random walk with drift)."""
    naive = np.stack([np.full(horizon, s[-1]) for s in train])
    snaive = np.stack([np.resize(s[-m:], horizon) if len(s) >= m else np.full(horizon, s[-1]) for s in train])
    steps = np.arange(1, horizon + 1)
    drift = np.stack([s[-1] + steps * (s[-1] - s[0]) / max(len(s) - 1, 1) for s in train])
    return {'Naive': naive, 'Seasonal naive': snaive, 'Drift': drift}


def mase_scale(train, m):
    """In-sample mean absolute seasonal difference per series (M4 definition)."""
    return np.array([np.mean(np.abs(s[m:] - s[:-m])) if len(s) > m else np.nan for s in train])


def smape(forecast, actual):
    """Per series, in percent."""
    return np.mean(200 * np.abs(forecast - actual) / (np.abs(forecast) + np.abs(actual)), axis=1)


def mase(forecast, actual, scale):
    return np.mean(np.abs(forecast - actual), axis=1) / scale


def msis(lower, upper, actual, scale, alpha=0.05):
    penalty = (upper - lower + 2 / alpha * (lower - actual) * (actual < lower)
               + 2 / alpha * (actual - upper) * (actual > upper))
    return np.mean(penalty, axis=1) / scale


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a trained PFN on M4.')
    parser.add_argument('--checkpoint', required=True, help='Run folder or model.pt.')
    parser.add_argument('--frequency', default='Yearly', choices=list(M4_FREQUENCIES))
    parser.add_argument('--data_dir', default='datasets/m4')
    parser.add_argument('--max_series', type=int, default=None, help='Use only the first N series.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_predictions', default=None,
                        help='CSV file for the PFN forecasts (id, step, mean, lower_95, upper_95, actual).')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    horizon, m = M4_FREQUENCIES[args.frequency]
    model, config, _ = load_checkpoint(args.checkpoint, device=args.device)
    bptt = config.training.bptt
    max_context = bptt - horizon
    if max_context < 1:
        raise ValueError(f'bptt={bptt} leaves no room for context with horizon {horizon}.')

    ids, train, test = load_m4(args.data_dir, args.frequency)
    if args.max_series is not None:
        ids, train, test = ids[:args.max_series], train[:args.max_series], test[:args.max_series]
    actual = np.stack(test)
    lengths = np.array([len(s) for s in train])
    print(f'{args.frequency}: {len(train)} series, horizon {horizon}, training lengths {lengths.min()}..'
          f'{lengths.max()} (median {int(np.median(lengths))}); context = last <= {max_context} points '
          f'({(lengths > max_context).mean():.0%} of the series are truncated)')

    with torch.no_grad():
        mean, lower, upper, predictions, loc, scale_std = pfn_forecast(
            model, train, horizon, max_context, args.batch_size)

    scale = mase_scale(train, m)
    valid = np.isfinite(scale) & (scale > 0)
    if not valid.all():
        print(f'Skipping {(~valid).sum()} series with a MASE scale of 0 or undefined.')

    forecasts = {'PFN (mean)': mean, **naive_forecasts(train, horizon, m)}
    reference = 'Naive' if m == 1 else 'Seasonal naive'
    ref_smape = smape(forecasts[reference], actual)[valid].mean()
    ref_mase = mase(forecasts[reference], actual, scale)[valid].mean()

    print(f'\n{"method":16s} {"sMAPE":>8s} {"MASE":>8s} {"OWA":>8s}')
    for name, fc in forecasts.items():
        s, q = smape(fc, actual)[valid].mean(), mase(fc, actual, scale)[valid].mean()
        print(f'{name:16s} {s:8.3f} {q:8.3f} {0.5 * (s / ref_smape + q / ref_mase):8.3f}')
    print(f'(OWA relative to {reference})')

    actual_std = torch.tensor((actual - loc) / scale_std)
    nll = torch.nn.functional.gaussian_nll_loss(
        torch.stack([p for p, _ in predictions]).double(), actual_std,
        torch.stack([v for _, v in predictions]).double(), full=True, reduction='none')
    covered = (actual >= lower) & (actual <= upper)
    print(f'\nPFN 95% interval: coverage {covered.mean():.3f}, '
          f'MSIS {msis(lower, upper, actual, scale)[valid].mean():.3f}')
    print(f'PFN Gaussian NLL (standardised): mean {nll.mean().item():.3f}, median {nll.median().item():.3f}')
    print('Coverage per horizon step: ' + ' '.join(f'{c:.2f}' for c in covered.mean(0)))

    if args.save_predictions:
        with open(args.save_predictions, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'step', 'mean', 'lower_95', 'upper_95', 'actual'])
            for i, series_id in enumerate(ids):
                for h in range(horizon):
                    writer.writerow([series_id, h + 1, f'{mean[i, h]:.6g}', f'{lower[i, h]:.6g}',
                                     f'{upper[i, h]:.6g}', f'{actual[i, h]:.6g}'])
        print(f'Saved forecasts to {args.save_predictions}')


if __name__ == '__main__':
    main()
