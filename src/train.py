"""Minimal PFN training on the GP prior (`src/prior.py`).

Refactored copy of `train.py` with fixed choices:
    encoder:      nn.Linear(num_features, emsize)
    y_encoder:    nn.Linear(1, emsize)
    (no positional encoding: the model is invariant to the order of the context points)
    criterion:    `model.criterion` = nn.GaussianNLLLoss(reduction='none', full=True); the model returns
                  (mean, variance) for every query point (variance mapped to be positive inside the model)

All settings are defined in `src/config.py` (`TrainConfig`, the single source of default values).
Run from the repository root, e.g.:
    python -m src.train --config configs/linear_gp.yaml training.epochs=10 model.emsize=128 --output_dir outputs/run1
    python -m src.train --config configs/linear_gp.yaml --print_config   # show the resolved config and exit

The run folder `--output_dir` (required) contains `config.yaml` (resolved config, reusable with `--config`),
`metadata.yaml` (git commit, library versions, final losses), `model.pt` (see `load_checkpoint`) and
`tensorboard/` (view with `tensorboard --logdir outputs`; disable with `training.tensorboard=false`).
With `training.checkpoint_every=n`, `checkpoints/model_epoch_XXXX.pt` is also written every n epochs (same format).
"""
import argparse
import datetime
import os
import platform
import random
import subprocess
import time

import gpytorch
import torch
import yaml
from torch import nn

from src import prior
from src.config import TrainConfig, config_from_dict, config_to_dict, load_config, save_config
from src.transformer import GaussianTransformerModel
from src.utils import (
    default_device,
    get_cosine_schedule_with_warmup,
    get_openai_lr,
    get_uniform_single_eval_pos_sampler,
    get_weighted_single_eval_pos_sampler,
)


def build_model(config: TrainConfig, decoder=None):
    """Create the (untrained) `GaussianTransformerModel` described by `config`."""
    m = config.model

    return GaussianTransformerModel(
        encoder=nn.Linear(config.prior.num_features, m.emsize),
        ninp=m.emsize,
        nhead=m.nhead,
        nhid=m.nhid or 2 * m.emsize,
        nlayers=m.nlayers,
        dropout=m.dropout,
        y_encoder=nn.Linear(1, m.emsize),
        decoder=decoder,
        input_normalization=m.input_normalization,
        min_var=m.min_var)


def ensure_seed(config):
    """Draw a random `config.training.seed` (in place) if it is None."""
    if config.training.seed is None:
        config.training.seed = random.SystemRandom().randrange(2 ** 31)


def train(
        config: TrainConfig | None = None,
        priordataloader_class=prior.DataLoader,
        decoder=None,
        scheduler=get_cosine_schedule_with_warmup,
        load_weights_from_this_state_dict=None,
        verbose=True,
        checkpoint_fn=None,
        epoch_fn=None
):
    """Train a PFN with Gaussian NLL loss.

    :param config: all (serializable) settings, see `src/config.py`; default `TrainConfig()`.
        If `config.training.seed` is None, a random seed is drawn and written into `config` (in place),
        so the config always records the seed that was used.
    :param priordataloader_class: DataLoader class of the prior, constructed with `config.prior`
        (default: GP prior from `src/prior.py`)
    :param decoder: optional decoder factory `decoder(ninp, nhid, n_out)` (not part of the config)
    :param scheduler: factory `scheduler(optimizer, warmup_epochs, epochs)`, stepped once per epoch
    :param checkpoint_fn: optional `checkpoint_fn(epoch, model, total_loss, total_positional_losses)`, called
        after every `config.training.checkpoint_every` epochs except the last one (the returned model is final)
    :param epoch_fn: optional `epoch_fn(epoch, metrics)` called after every epoch, e.g. `TensorBoardLogger.log_epoch`;
        `metrics`: `loss`, `positional_losses` (list, nan = position not sampled), `lr`, `epoch_time`,
        `data_time`, `step_time`, `forward_time` (the last three: of the last batch)
    :return: (mean loss of the last epoch, mean loss per single_eval_pos of the last epoch, model on CPU)
    """
    config = config or TrainConfig()
    config_from_dict(config_to_dict(config))  # re-check all constraints (incl. cross-section) after in-place edits
    p, o, t = config.prior, config.optim, config.training

    ensure_seed(config)
    # The single_eval_pos samplers use python's `random`, everything else uses torch.
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    print(f'Using seed {t.seed}')

    device = t.device or default_device
    print(f'Using {device} device')

    bptt = t.bptt
    steps_per_epoch = t.steps_per_epoch
    epochs = t.epochs
    aggregate_k_gradients = o.aggregate_k_gradients
    get_sampler = {'weighted': get_weighted_single_eval_pos_sampler,
                   'uniform': get_uniform_single_eval_pos_sampler}[t.eval_pos_sampling]
    single_eval_pos_gen = get_sampler(t.max_eval_pos or bptt)

    prior_kwargs = {'num_features': p.num_features,
                    'equidistant_x': p.equidistant_x,
                    'sampling_method': p.sampling_method,
                    'basis_tol': p.basis_tol,
                    'forecast_prob': p.forecast_prob,
                    'hyperparameters': p.hyperparameters}
    if p.device is not None:
        prior_kwargs['device'] = p.device
    dl = priordataloader_class(
        num_steps=steps_per_epoch,
        batch_size=t.batch_size,
        seq_len=bptt,
        **prior_kwargs)

    model = build_model(config, decoder=decoder)

    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    model.to(device)

    lr = o.lr
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = scheduler(optimizer, o.warmup_epochs, epochs)

    def train_epoch():
        model.train()
        total_loss = 0.
        total_positional_losses = torch.zeros(bptt)
        total_positional_losses_recorded = torch.zeros(bptt)
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, \
            'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'

        for batch, ((x, y), targets) in enumerate(dl):
            time_to_get_batch = time.time() - before_get_batch
            before_forward = time.time()

            single_eval_pos = single_eval_pos_gen()
            assert 0 <= single_eval_pos < bptt, f'single_eval_pos must be in [0, {bptt}), got {single_eval_pos}.'

            # each (bptt - single_eval_pos, batch_size)
            mean_pred, var_pred = model((x.to(device), y.to(device)), single_eval_pos=single_eval_pos)
            forward_time = time.time() - before_forward

            targets = targets[single_eval_pos:].to(device)
            losses = model.criterion(mean_pred, targets, var=var_pred)  # (num_queries, batch_size)

            loss = losses.mean()
            loss.backward()
            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()

            step_time = time.time() - before_forward

            total_loss += loss.item()
            total_positional_losses[single_eval_pos] += loss.item()
            total_positional_losses_recorded[single_eval_pos] += 1

            before_get_batch = time.time()

        # positions that were never sampled in this epoch are nan
        positional_losses = (total_positional_losses / total_positional_losses_recorded).tolist()
        return total_loss / steps_per_epoch, positional_losses, time_to_get_batch, forward_time, step_time

    total_loss, total_positional_losses = float('inf'), None
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time = train_epoch()
        epoch_time = time.time() - epoch_start_time
        lr_epoch = scheduler.get_last_lr()[0]

        if verbose:
            print('-' * 89)
            print(
                f'| end of epoch {epoch:3d} | time: {epoch_time:5.2f}s | mean loss {total_loss:5.2f} | '
                f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {lr_epoch}"
                f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                f' forward time {forward_time:5.2f}')
            print('-' * 89)

        if epoch_fn is not None:
            epoch_fn(epoch, {'loss': total_loss, 'positional_losses': total_positional_losses, 'lr': lr_epoch,
                             'epoch_time': epoch_time, 'data_time': time_to_get_batch, 'step_time': step_time,
                             'forward_time': forward_time})

        scheduler.step()

        checkpoint_every = t.checkpoint_every or epochs
        if checkpoint_fn is not None and epoch % checkpoint_every == 0 and epoch < epochs:
            checkpoint_fn(epoch, model, total_loss, total_positional_losses)

    return total_loss, total_positional_losses, model.to('cpu')


def run_metadata():
    """Information about the environment of a run (for reproducibility)."""

    def git(*args):
        try:
            out = subprocess.run(['git', *args], capture_output=True, text=True, timeout=10,
                                 cwd=os.path.dirname(os.path.abspath(__file__)))
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None  # e.g. no git / no repository (Docker image)

    status = git('status', '--porcelain')
    return {
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git('rev-parse', 'HEAD'),
        'git_dirty': None if status is None else bool(status),
        'python': platform.python_version(),
        'torch': str(torch.__version__),  # TorchVersion (str subclass) is not YAML-serializable
        'gpytorch': str(gpytorch.__version__),
        'cuda': torch.version.cuda,
    }


def flatten_config(config):
    """Flat `{'prior.hyperparameters.noise.low': 1e-4, ...}` with TensorBoard-compatible values (None -> 'null')."""

    def walk(value, prefix):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from walk(v, f'{prefix}.{k}' if prefix else k)
        else:
            yield prefix, 'null' if value is None else value

    return dict(walk(config_to_dict(config), ''))


class TensorBoardLogger:
    """Write the config (Text and HParams tabs) and the per-epoch losses (Scalars tab) of a run to `log_dir`.

    View with `tensorboard --logdir outputs` (all runs) or `--logdir outputs/run5/tensorboard` (one run).
    """

    def __init__(self, log_dir, config, metadata=None):
        from torch.utils.tensorboard import SummaryWriter
        from torch.utils.tensorboard.summary import hparams

        self.writer = SummaryWriter(log_dir=log_dir)

        def as_text(data):  # markdown code block (indented)
            return '\n'.join('    ' + line for line in yaml.safe_dump(data, sort_keys=False).splitlines())

        self.writer.add_text('config', as_text(config_to_dict(config)))
        if metadata:
            self.writer.add_text('metadata', as_text(metadata))
        # Session start now and end in `close`, so running (or interrupted) runs appear in the HParams tab too;
        # the metric column shows the latest value of the `loss/mean` scalar.
        exp, ssi, self._sei = hparams(flatten_config(config), {'loss/mean': None})
        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.flush()

    def log_epoch(self, epoch, metrics):
        w = self.writer
        w.add_scalar('loss/mean', metrics['loss'], epoch)
        for pos, loss in enumerate(metrics['positional_losses']):
            if loss == loss:  # skip nan (context size not sampled in this epoch)
                w.add_scalar(f'loss_by_context_size/{pos:04d}', loss, epoch)
        w.add_scalar('optim/lr', metrics['lr'], epoch)
        for name in ('epoch_time', 'data_time', 'step_time', 'forward_time'):
            w.add_scalar(f'time/{name}', metrics[name], epoch)
        w.flush()

    def close(self):
        self.writer.file_writer.add_summary(self._sei)
        self.writer.close()


def save_checkpoint(path, config, model, metadata):
    """Write model weights (on CPU), config and metadata to `path` (readable by `load_checkpoint`)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config': config_to_dict(config),
                'metadata': metadata}, path)


def save_intermediate_checkpoint(output_dir, config, metadata, epoch, model, total_loss, total_positional_losses):
    """Write `output_dir/checkpoints/model_epoch_XXXX.pt` (same format as the final `model.pt`)."""
    path = os.path.join(output_dir, 'checkpoints', f'model_epoch_{epoch:04d}.pt')
    save_checkpoint(path, config, model, {**metadata, 'epoch': epoch, 'total_loss': total_loss,
                                          'total_positional_losses': total_positional_losses})
    print(f'Saved checkpoint to {path}')


def save_run(output_dir, config, model, total_loss, total_positional_losses, metadata):
    """Write `config.yaml`, `metadata.yaml` and `model.pt` to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = {**metadata, 'epoch': config.training.epochs, 'total_loss': total_loss,
                'total_positional_losses': total_positional_losses}
    # model.pt first: it is the most valuable file if anything below fails.
    save_checkpoint(os.path.join(output_dir, 'model.pt'), config, model, metadata)
    save_config(config, os.path.join(output_dir, 'config.yaml'))
    with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    print(f'Saved run to {output_dir}')


def load_checkpoint(path, device='cpu'):
    """Load a model saved by `save_run` (`path`: run folder or its `model.pt`).

    :return: (model in eval mode, `TrainConfig`, metadata dict)
    """
    if os.path.isdir(path):
        path = os.path.join(path, 'model.pt')
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = config_from_dict(checkpoint['config'])
    model = build_model(config)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), config, checkpoint['metadata']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train a PFN on the GP prior. Settings: defaults in src/config.py <- --config files '
                    '<- dotted overrides.')
    parser.add_argument('--config', action='append', default=[],
                        help='YAML config file; can be given several times (merged in order).')
    parser.add_argument('--output_dir',
                        help='Run folder for config.yaml, metadata.yaml, model.pt, checkpoints/ and tensorboard/; '
                             'must not contain a model.pt. Required unless --print_config is given.')
    parser.add_argument('--print_config', action='store_true', help='Print the resolved config and exit.')
    parser.add_argument('overrides', nargs='*', metavar='key.sub=value',
                        help='Override single values, e.g. model.emsize=256 prior.hyperparameters.noise=0.01')
    args = parser.parse_intermixed_args(argv)
    if args.output_dir is None and not args.print_config:
        parser.error('the following arguments are required: --output_dir')
    return args


if __name__ == '__main__':
    args = parse_args()
    config = load_config(*args.config, overrides=args.overrides)

    if args.print_config:
        print(yaml.safe_dump(config_to_dict(config), sort_keys=False), end='')
        raise SystemExit

    # Fail before training rather than after it.
    if os.path.exists(os.path.join(args.output_dir, 'model.pt')):
        raise FileExistsError(f'{args.output_dir} already contains a model.pt; choose another --output_dir.')

    print('Config:\n' + yaml.safe_dump(config_to_dict(config), sort_keys=False))
    metadata = run_metadata()

    def checkpoint_fn(epoch, model, total_loss, total_positional_losses):
        save_intermediate_checkpoint(args.output_dir, config, metadata, epoch, model, total_loss,
                                     total_positional_losses)

    logger = None
    if config.training.tensorboard:
        ensure_seed(config)  # so that the logged config contains the seed
        logger = TensorBoardLogger(os.path.join(args.output_dir, 'tensorboard'), config, metadata)
    try:
        total_loss, total_positional_losses, model = train(
            config, checkpoint_fn=checkpoint_fn, epoch_fn=logger and logger.log_epoch)
    finally:
        if logger is not None:
            logger.close()

    save_run(args.output_dir, config, model, total_loss, total_positional_losses, metadata)
