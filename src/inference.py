"""Batched inference for datasets with different numbers of context and query points.

Datasets of different lengths are padded to a common length and the padded tokens are masked per dataset
(`BaseTransformerModel.forward(..., src_key_padding_mask=...)`), so each prediction is the same as when the
dataset is run on its own:

    [ padding | context_i ]  [ queries_i | padding ]
     <-- single_eval_pos -->  <-- num_queries ---->

The model has no positional encoding, so the position of the padding does not matter; context points are
left-padded and queries right-padded so that the real values sit next to the context/query boundary.

Run from the repository root to try it on synthetic data drawn from the GP prior (forecasting the last
points of equidistant series, like yearly M4 series):
    python -m src.inference --checkpoint outputs/run1
    python -m src.inference --num_series 64 --horizon 3     # untrained model from the default config
"""
import argparse
import random

import torch

from src import prior
from src.config import GPHyperparameters, TrainConfig
from src.train import build_model, load_checkpoint


def pad_datasets(contexts, queries):
    """Pad datasets of different sizes into one batch.

    :param contexts: list of `(x, y)` per dataset, x: (n_i, num_features), y: (n_i,); n_i may be 0
    :param queries: list of query inputs per dataset, x: (m_i, num_features), m_i >= 1
    :return: dict with
        - `x`: (seq_len, batch_size, num_features), `y`: (seq_len, batch_size) (0 at padded and query positions),
        - `single_eval_pos`: max n_i (number of context positions),
        - `src_key_padding_mask`: bool (batch_size, seq_len), True = padding,
        - `query_mask`: bool (max m_i, batch_size), True = real query
    """
    assert len(contexts) == len(queries) > 0, 'contexts and queries must be non-empty lists of the same length.'
    num_features = queries[0].shape[-1]
    num_context = max(len(x) for x, _ in contexts)
    num_queries = max(len(x) for x in queries)
    assert min(len(x) for x in queries) > 0, 'every dataset needs at least one query point.'
    batch_size, seq_len = len(contexts), num_context + num_queries
    ref = queries[0]

    x = ref.new_zeros(seq_len, batch_size, num_features)
    y = ref.new_zeros(seq_len, batch_size)
    padding = torch.ones(batch_size, seq_len, dtype=torch.bool, device=ref.device)
    for b, ((x_c, y_c), x_q) in enumerate(zip(contexts, queries)):
        n, m = len(x_c), len(x_q)
        x[num_context - n:num_context, b] = x_c
        y[num_context - n:num_context, b] = y_c
        x[num_context:num_context + m, b] = x_q
        padding[b, num_context - n:num_context + m] = False

    return {'x': x, 'y': y, 'single_eval_pos': num_context, 'src_key_padding_mask': padding,
            'query_mask': ~padding[:, num_context:].T}


@torch.no_grad()
def predict(model, contexts, queries, batch_size=256):
    """Predict all query points of datasets with different sizes in padded batches.

    :param model: trained PFN in eval mode (e.g. from `load_checkpoint`); data are moved to its device
    :param contexts, queries: as in `pad_datasets`
    :return: list with `(mean, var)` per dataset on the CPU, each of shape (m_i,)
    """
    device = next(model.parameters()).device
    results = []
    for start in range(0, len(contexts), batch_size):
        batch = pad_datasets([(x.to(device), y.to(device)) for x, y in contexts[start:start + batch_size]],
                             [x.to(device) for x in queries[start:start + batch_size]])
        mean, var = model((batch['x'], batch['y']), single_eval_pos=batch['single_eval_pos'],
                          src_key_padding_mask=batch['src_key_padding_mask'])
        for b, x_q in enumerate(queries[start:start + batch_size]):
            results.append((mean[:len(x_q), b].cpu(), var[:len(x_q), b].cpu()))
    return results


def generate_forecasting_data(num_series, min_context, max_context, horizon,
                              hyperparameters: GPHyperparameters | dict | None = None, equidistant_x=True):
    """Synthetic forecasting tasks from the GP prior (`prior.get_batch`), one per series with its own length.

    Series i has `n_i + horizon` points (`n_i` uniform in [min_context, max_context]) with x in [0, 1]
    (equidistant or uniform, sorted); the first n_i points are the context, the last `horizon` are the queries.

    :return: `contexts` (list of `(x, y)`), `queries` (list of x), `targets` (list of y at the queries)
    """
    contexts, queries, targets = [], [], []
    for _ in range(num_series):
        n = random.randint(min_context, max_context)
        x, y, _ = prior.get_batch(1, n + horizon, 1, device='cpu', hyperparameters=hyperparameters,
                                  equidistant_x=equidistant_x)
        order = x[:, 0, 0].argsort()  # forecasting: queries are the largest x
        x, y = x[order, 0], y[order, 0]  # (n + horizon, 1), (n + horizon,)
        contexts.append((x[:n], y[:n]))
        queries.append(x[n:])
        targets.append(y[n:])
    return contexts, queries, targets


def gaussian_metrics(predictions, targets):
    """Mean Gaussian NLL and coverage of the central 90% interval over all query points."""
    mean = torch.cat([m for m, _ in predictions])
    var = torch.cat([v for _, v in predictions])
    target = torch.cat(targets)
    nll = torch.nn.functional.gaussian_nll_loss(mean, target, var, full=True).item()
    coverage = ((target - mean).abs() <= 1.6449 * var.sqrt()).float().mean().item()
    return nll, coverage


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Batched (padded) PFN inference on synthetic GP data.')
    parser.add_argument('--checkpoint', default=None,
                        help='Run folder or model.pt; default: untrained model from the default config.')
    parser.add_argument('--num_series', type=int, default=200)
    parser.add_argument('--horizon', type=int, default=None,
                        help='Query points per series; default: min(6, bptt // 2).')
    parser.add_argument('--min_context', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.checkpoint is not None:
        model, config, _ = load_checkpoint(args.checkpoint)
    else:
        config = TrainConfig()
        model = build_model(config).eval()
    if config.prior.num_features != 1:
        raise ValueError('The demo generates 1D series; the model needs num_features == 1.')

    # The model has seen at most bptt - 1 context points and bptt points in total.
    bptt = config.training.bptt
    horizon = args.horizon or min(6, bptt // 2)
    max_context = bptt - horizon
    assert 0 <= args.min_context <= max_context, f'min_context must be in [0, {max_context}].'

    contexts, queries, targets = generate_forecasting_data(
        args.num_series, args.min_context, max_context, horizon,
        hyperparameters=config.prior.hyperparameters, equidistant_x=config.prior.equidistant_x)
    print(f'{args.num_series} series, context sizes {args.min_context}..{max_context}, horizon {horizon}')

    model.to(args.device)
    batched = predict(model, contexts, queries, batch_size=args.batch_size)
    single = predict(model, contexts, queries, batch_size=1)  # no padding at all
    max_diff = max(max((mb - ms).abs().max().item(), (vb - vs).abs().max().item())
                   for (mb, vb), (ms, vs) in zip(batched, single))
    print(f'max |padded batch - one by one| (mean, var): {max_diff:.2e}')

    # The same padded batch without the padding mask: padded (x=0, y=0) tokens act as observations.
    unmasked = []
    for start in range(0, len(contexts), args.batch_size):
        batch = pad_datasets(contexts[start:start + args.batch_size], queries[start:start + args.batch_size])
        with torch.no_grad():
            mean, var = model((batch['x'].to(args.device), batch['y'].to(args.device)),
                              single_eval_pos=batch['single_eval_pos'])
        unmasked += [(mean[:len(x_q), b].cpu(), var[:len(x_q), b].cpu())
                     for b, x_q in enumerate(queries[start:start + args.batch_size])]
    max_diff_unmasked = max((mu - ms).abs().max().item() for (mu, _), (ms, _) in zip(unmasked, single))
    print(f'max |padded batch without mask - one by one| (mean): {max_diff_unmasked:.2e}')

    for name, predictions in [('padded + mask', batched), ('padded, no mask', unmasked)]:
        nll, coverage = gaussian_metrics(predictions, targets)
        print(f'{name:16s} NLL {nll:7.3f} | 90% interval coverage {coverage:.3f}')


if __name__ == '__main__':
    main()
