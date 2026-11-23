# `src/`: training a Prior-Fitted Network (PFN) on a Gaussian-process prior

This package trains a transformer to do **approximate Bayesian inference for regression in a single forward
pass**. The approach comes from *Transformers Can Do Bayesian Inference* (Müller et al., ICLR 2022,
[arXiv:2112.10510](https://arxiv.org/abs/2112.10510)). The code started as a minimal, refactored subset of the
paper's code ([automl/TransformersCanDoBayesianInference](https://github.com/automl/TransformersCanDoBayesianInference),
Apache-2.0), restricted to 1D regression with Gaussian-process priors and a Gaussian output. See
"Origin, license and citation" at the end.

## The idea in short

1. **Prior.** Define a prior over functions, here a Gaussian process (GP) whose kernel hyperparameters can be
   random themselves.
2. **Synthetic data.** Repeatedly sample a function from the prior, evaluate it with noise at `bptt` random
   inputs `x`, and split the resulting sequence into
   - *training (context) points* `(x_i, y_i)`, `i < single_eval_pos`, and
   - *query points* `x_j`, `j >= single_eval_pos`, whose `y_j` the model must predict.
3. **Training.** The transformer sees the context points and the query inputs, and outputs a predictive
   distribution (here a Gaussian `N(mean, var)`) for every query point. It is trained by minimising the
   negative log-likelihood of the true `y_j`.

Minimising this loss over many sampled datasets makes the network approximate the **posterior predictive
distribution** `p(y | x, D)` of the prior. After training, new data only needs one forward pass: no
MCMC, no hyperparameter fitting, no matrix inversion.

## Files

| File | Content |
|---|---|
| `config.py` | All settings as typed, validated pydantic models (`TrainConfig`), YAML loading/saving, command-line overrides. The single source of default values. |
| `prior.py` | The GP prior: hyperparameter sampling, batched kernel, `get_batch` and the `DataLoader` used for training. |
| `transformer.py` | The PFN transformer (`BaseTransformerModel`) with a Gaussian output head (`GaussianTransformerModel`). Also contains a Beta-Binomial head that is not yet used for training. |
| `train.py` | Training loop, saving/loading of runs and the command-line interface (`python -m src.train`). |
| `inference.py` | Batched inference for datasets of different sizes (padding + per-dataset mask: `pad_datasets`, `predict`) and a demo on synthetic forecasting data (`python -m src.inference`). |
| `evaluate_m4.py` | Evaluation on the M4 competition data in `datasets/m4/` (not in git, see "Evaluation on M4"): sMAPE, MASE, OWA, MSIS against naive/drift baselines (`python -m src.evaluate_m4 --checkpoint DIR --frequency Yearly`). |
| `download_m4.py` | Downloads the M4 CSV files from github.com/Mcompetitions/M4-methods (folder `Dataset/`) into `datasets/m4/` (`python -m src.download_m4`). |
| `utils.py` | Device selection, learning-rate schedule, `single_eval_pos` samplers, `SeqBN`, and `get_batch_to_dataloader` (turns a `get_batch` function into a data loader). |
| `gp_benchmark.py` | Comparison with the exact GP posterior on the stored benchmark `benchmarks/gp_benchmark_v1.npz` (fixed hyperparameter subsets): KL, NLL gap, error of mean and variance (`python -m src.gp_benchmark evaluate --checkpoint DIR`). |

## Project layout

| Folder | Content | In git |
|---|---|---|
| `src/` | The package (this README and `LICENSE`). | yes |
| `configs/` | Example training configs (YAML), see "Example configs". | yes |
| `benchmarks/` | `gp_benchmark_v1.npz`, the exact-GP benchmark (about 2 MB). | yes |
| `datasets/m4/` | M4 competition data for `evaluate_m4.py` (`python -m src.download_m4`). | no |
| `outputs/` | Run folders written by `python -m src.train --output_dir outputs/...`. | no |

## Setup

All commands are run from the **repository root** (the folder that contains `src/`, `configs/`, `benchmarks/`,
`pyproject.toml`). Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # creates .venv with the locked dependencies (Python 3.12)
source .venv/bin/activate    # or prefix commands with `uv run`
```

Main dependencies: `torch`, `gpytorch`/`linear_operator` (stable Cholesky), `pydantic` and `omegaconf`
(configuration), `pyyaml`.

## Quick start

```bash
# Show the fully resolved configuration (defaults + YAML + overrides) and exit
python -m src.train --config configs/gp_hyperprior.yaml --print_config

# Train and save the run
python -m src.train --config configs/gp_hyperprior.yaml training.epochs=10 --output_dir outputs/my_run

# Small, fast smoke test on the CPU
python -m src.train --config configs/linear_gp.yaml training.epochs=2 training.steps_per_epoch=2 \
    training.batch_size=8 model.emsize=64 model.nlayers=1 prior.device=cpu training.device=cpu \
    --output_dir outputs/smoke_test
```

Command-line arguments:

- `--config FILE`: a YAML file. It can be given several times; files are merged in order.
- `key.sub=value`: overrides a single value (parsed as YAML), e.g. `model.emsize=256`,
  `optim.lr=null`, or `prior.hyperparameters.noise={dist: uniform, low: 0.01, high: 0.1}`.
- `--output_dir DIR` (required unless `--resume` or `--print_config` is given): the run folder. It must not
  already contain a `model.pt` (finished run) or a `checkpoints/last.pt` (interrupted run); this is checked
  before training starts.
- `--resume DIR`: continue the interrupted run in `DIR` (see "Resuming an interrupted run"). It cannot be
  combined with `--config`, overrides or `--output_dir`.
- `--print_config`: print the resolved config and exit.

The device is chosen automatically (CUDA, then Apple MPS, then CPU) unless `training.device` or
`prior.device` is set. **MPS has no float64**, so sampling the GP prior with the Cholesky method on MPS can be
numerically unstable for very small noise; use `prior.device=cpu` if you see "not p.d." warnings or errors.

## Configuration (`config.py`)

A configuration is assembled as follows (later wins):

1. YAML files given with `--config` (they only need the values that differ from the defaults),
2. dotted overrides `key.sub=value`.

Pydantic then validates the result and fills in the defaults of `TrainConfig` for anything missing.
Unknown keys, wrong types and violated constraints raise a `pydantic.ValidationError` (a `ValueError`)
that names every problem with its path, e.g. `model.emsize (100) must be divisible by nhead (3)`.

```python
from src.config import load_config, save_config, TrainConfig

cfg = load_config('configs/gp_hyperprior.yaml', overrides=['training.epochs=5'])
cfg.model.emsize = 256          # assignments are validated too; invalid values are rolled back
default = TrainConfig()         # all defaults
save_config(cfg, 'my_config.yaml')
```

Constraints that span two sections, such as `training.steps_per_epoch % optim.aggregate_k_gradients == 0`,
are not checked on assignment. They are checked by `load_config`, `config_from_dict` and at the start of
`train`.

### Sections

| Section | Field | Default | Meaning |
|---|---|---|---|
| `prior` | `num_features` | 1 | Dimension of `x`. |
| | `equidistant_x` | false | `x` on an equidistant grid in [0, 1] instead of uniform random; requires `num_features == 1`. |
| | `device` | null | Device for sampling the prior (null = automatic). |
| | `sampling_method` | auto | `basis` (only `num_features == 1`; fast, see below), `cholesky` (exact, any `num_features`), `auto` = `basis` if `num_features == 1`, else `cholesky`. |
| | `basis_tol` | 1e-4 | Accuracy of the `basis` method, in [1e-6, 0.1]. |
| | `forecast_prob` | 0 | Fraction of datasets whose `x` is sorted, so that the context holds the smallest and the queries the largest `x` (forecasting instead of interpolation); requires `num_features == 1`; no effect with `equidistant_x` (already sorted). |
| | `hyperparameters` | see below | GP hyperparameters. |
| `model` | `emsize` | 512 | Embedding size; must be divisible by `nhead`. |
| | `nhid` | null | Hidden size of the feed-forward layers and the decoder MLP (null = `2 * emsize`). |
| | `nlayers` | 6 | Number of transformer layers. |
| | `nhead` | 4 | Number of attention heads. |
| | `dropout` | 0.0 | In [0, 1). |
| | `input_normalization` | false | Batch-normalise the input tokens (`SeqBN`). |
| | `min_var` | 1e-6 | Lower bound of the predicted variance. |
| `optim` | `lr` | 0.001 | Adam learning rate (null = heuristic `get_openai_lr` based on the model size). |
| | `warmup_epochs` | 50 | Linear warm-up, then cosine decay (stepped once per epoch). |
| | `aggregate_k_gradients` | 1 | Gradient accumulation over this many steps. |
| `training` | `epochs` | 200 | |
| | `steps_per_epoch` | 10 | Batches per epoch (each is freshly sampled from the prior). |
| | `batch_size` | 1000 | Datasets (sequences) per batch. |
| | `bptt` | 10 | Sequence length = number of context + query points per dataset; >= 2. |
| | `max_eval_pos` | null | `single_eval_pos` is drawn from `range(max_eval_pos)` (null = `bptt`). |
| | `eval_pos_sampling` | weighted | `weighted`: position `i` has weight `1 / (max_eval_pos - i)`, so every query point is equally likely to be seen with any context size. `uniform`: all positions equally likely. |
| | `seed` | null | null = a random seed is drawn and written into the saved config. |
| | `device` | null | Device for the model (null = automatic). |
| | `checkpoint_every` | null | Save `checkpoints/model_epoch_XXXX.pt` in the run folder every n epochs. null = `epochs`, i.e. only the final `model.pt`. |
| | `tensorboard` | true | Write TensorBoard logs (losses, learning rate, config) to `<output_dir>/tensorboard/`. |
| | `gp_benchmark` | null | Path of an exact-GP benchmark file (e.g. `benchmarks/gp_benchmark_v1.npz`) evaluated during training; the results are printed and logged to TensorBoard. null = no evaluation. |
| | `gp_benchmark_every` | null | Evaluate `gp_benchmark` every n epochs and after the last epoch. null = `epochs`, i.e. only after the last epoch. |

### GP hyperparameters (`prior.hyperparameters`)

The kernel of the prior is

```
k(x, x') = bias
         + variance             * x^T x'                                   (random linear trend)
         + rbf_outputscale      * exp(-||x - x'||^2 / (2 rbf_lengthscale^2))  (smooth deviations)
         + periodic_outputscale * exp(-2 sum_i sin^2(pi |x_i - x'_i| / period) / periodic_lengthscale^2)
                                * [exp(-||x - x'||^2 / (2 periodic_decay^2))]  (optional factor)
```

and `y = f(x) + eps` with `eps ~ N(0, noise)`. The GP has zero mean.

| Field | Default | Meaning / constraint |
|---|---|---|
| `noise` | 0.1 | Observation noise variance; > 0. |
| `variance` | 0.1 | Variance of the slopes of the linear part; >= 0. |
| `bias` | 0 | Variance of the constant offset; >= 0 (0 = none). |
| `rbf_outputscale` | 0.1 | Amplitude² of the RBF part; >= 0 (0 = no RBF part). |
| `rbf_lengthscale` | 0.1 | > 0 if `rbf_outputscale` can be > 0. |
| `periodic_outputscale` | 0 | Amplitude² of the periodic part; >= 0 (0 = no periodic part). |
| `period` | null | > 0 if `periodic_outputscale` can be > 0. |
| `periodic_lengthscale` | null | Smoothness within one period (small = more harmonics); > 0 if the periodic part is on. Note: gpytorch's `PeriodicKernel` does not square its lengthscale, so its lengthscale equals `periodic_lengthscale²`. |
| `periodic_decay` | null | Lengthscale of the optional RBF factor that makes the periodic part *locally* periodic; null = exactly periodic. |

**Hyperpriors.** Every hyperparameter is either a fixed number or a distribution that is sampled
**independently for every dataset** in a batch, so the PFN learns to infer the hyperparameters from the
context points:

```yaml
noise:   { dist: log_uniform, low: 1.e-4, high: 0.1 }               # uniform in log space; low > 0
bias:    { dist: uniform, low: 0.0, high: 1.0 }
variance: { dist: log_uniform, low: 0.1, high: 10.0, zero_prob: 0.2 } # zero-inflated
```

With `zero_prob = p` the value is exactly 0 with probability `p`, and drawn from `dist` otherwise. This
switches a kernel part off for a fraction of the datasets. It is allowed for `variance`, `bias`,
`rbf_outputscale` and `periodic_outputscale`. Fields that must be positive (`noise`, lengthscales,
`period`, `periodic_decay`) reject it.

### Example configs (`configs/`)

- `linear_gp.yaml`: linear-trend GP on an equidistant 1D grid (only the differences from the defaults).
- `gp_hyperprior.yaml`: lists **every** setting, with hyperpriors (including zero-inflated ones) for all
  kernel parts, including the periodic part. Half of the datasets are forecasting tasks (`forecast_prob: 0.5`).
  A good template for new experiments.

## The prior (`prior.py`)

```python
from src.prior import get_batch

x, y, target_y = get_batch(batch_size=8, seq_len=20, num_features=1, device='cpu',
                           hyperparameters={'noise': 0.01, 'rbf_outputscale': 1.0, 'rbf_lengthscale': 0.2})
# x: (seq_len, batch_size, num_features), y == target_y: (seq_len, batch_size)
```

- `hyperparameters` can be a `GPHyperparameters` object, a dict (validated into one; missing keys take
  the defaults, unknown keys raise) or `None` (all defaults).
- Inputs are uniform in [0, 1]^num_features (or an equidistant grid with `equidistant_x=True`).
- With `forecast_prob=p`, the `x` values of each dataset are sorted with probability `p`. The split at
  `single_eval_pos` then turns into "past → future", so the model learns to forecast. Because there is no
  positional encoding, sorting does not make token positions matter, and the attention mask stays the same.
- `sample_hyperparameters(hp, batch_size)` returns one tensor of shape `(batch_size,)` per hyperparameter.
- `DataLoader = get_batch_to_dataloader(get_batch)` is a data loader that yields `num_steps` freshly
  sampled batches `((x, y), target_y)` per epoch.

**Sampling methods** (`get_batch(sampling_method=...)`, config `prior.sampling_method`):

| Method | How | Cost per dataset | Accuracy |
|---|---|---|---|
| `cholesky` | `dot_product_rbf_covariance(x, ...)` builds the `(batch_size, n, n)` covariance; samples via a jitter-safe Cholesky (`linear_operator.psd_safe_cholesky`). | O(n²) memory, O(n³) time | exact (jitter may raise the effective noise to ~1e-4 in float32) |
| `basis` (`num_features == 1`) | `sample_basis_1d`: each kernel part is sampled directly as a random combination of basis functions, without any matrix. | O(n) | see below |

How the `basis` method samples each part:
- Linear (`√variance·w·x`), bias (`√bias·z`) and noise: exact.
- Periodic: exact Fourier series (`periodic_fourier_coefficients`), truncated once the missing variance is below
  `basis_tol`. Small `periodic_lengthscale` values need more harmonics (about 8 for 0.5, about 40 for 0.1).
- RBF: Hilbert-space approximation (`rbf_basis_1d`; Solin & Särkkä, 2020). The data interval is padded so that
  boundary effects stay below `basis_tol`, and enough sine basis functions are used for the spectral
  truncation error to stay below `basis_tol` too (about 30 for `rbf_lengthscale=0.05`). The kernel error is about
  1–2 × `basis_tol`.
- Periodic × RBF (`periodic_decay`): every harmonic gets an independent RBF-process amplitude, which gives exactly
  the locally periodic kernel (up to the RBF approximation).

At `bptt=1024`, `batch_size=500` on a CPU, sampling a batch takes 0.02 s with `basis` and 6.5 s with `cholesky`.
The two methods produce different batches for the same seed. To reproduce a run trained before
`sampling_method` existed, set `prior.sampling_method=cholesky`.

**Custom priors.** Any function `get_batch(batch_size, seq_len, num_features, **kwargs) -> (x, y, target_y)`
with the shapes above can be wrapped with `src.utils.get_batch_to_dataloader` and passed to
`train(config, priordataloader_class=...)`.

## The model (`transformer.py`)

- `x` and `y` are embedded with linear layers (`nn.Linear(num_features, emsize)`, `nn.Linear(1, emsize)`).
  A context token is `enc(x_i) + enc(y_i)`; a query token is `enc(x_j)` only.
- A standard `nn.TransformerEncoder` (GELU) runs over all tokens with a special attention mask: every token
  attends to all context tokens and to itself, and query tokens do **not** attend to each other. Each
  prediction therefore depends only on the context and its own `x`. There is no positional encoding, so the
  context order does not matter.
- An MLP decoder maps each query token to two numbers. `GaussianTransformerModel` turns them into
  `mean` and `var = softplus(raw) + min_var`. Its loss is `model.criterion = nn.GaussianNLLLoss(full=True)`.
- `forward((x, y), single_eval_pos)` returns `(mean, var)`, each of shape
  `(seq_len - single_eval_pos, batch_size)`. The `y` values at query positions are ignored.
- Optional `src_key_padding_mask` (bool, `(batch_size, seq_len)`, True = padding): no token attends to a
  padded token, so datasets with different numbers of context/query points can share a batch and get the
  same predictions as on their own. `add_padding_to_mask` turns it into a per-dataset mask of shape
  `(batch_size * nhead, seq_len, seq_len)`. Padded tokens still attend to themselves, which keeps every
  attention row finite. Padded inputs must be finite (e.g. 0).

## Training (`train.py`)

`train(config)` builds the model and the prior data loader, then runs Adam with gradient clipping (norm 1)
and the warm-up + cosine learning-rate schedule. For every batch, a new `single_eval_pos` (the number of
context points) is drawn, so the model learns to predict from any context size between 0 and `bptt - 1`.
After each epoch it prints the mean loss and the loss per `single_eval_pos` (`nan` = position not sampled
in that epoch). It returns `(mean loss of last epoch, losses per position, model on CPU)`.
Optionally, `train(config, checkpoint_fn=f)` calls `f(epoch, model, loss, pos_losses)` after every
`training.checkpoint_every` epochs (not after the last one); the CLI uses this to write intermediate checkpoints.
`train(config, epoch_fn=g)` calls `g(epoch, metrics)` after every epoch (used for TensorBoard, see below).
`train(config, state_fn=h)` calls `h(epoch, state)` after every epoch with the full training state, and
`train(config, resume_state=state)` continues from such a state (used by `--resume`, see below).

```python
from src.config import load_config
from src.train import train, save_run, run_metadata, save_intermediate_checkpoint

cfg = load_config('configs/gp_hyperprior.yaml', overrides=['training.epochs=5', 'training.checkpoint_every=2'])
metadata = run_metadata()
loss, pos_losses, model = train(cfg, checkpoint_fn=lambda *a: save_intermediate_checkpoint(
    'outputs/notebook_run', cfg, metadata, *a))  # optional
save_run('outputs/notebook_run', cfg, model, loss, pos_losses, metadata)
```

### Run folders and checkpoints

`--output_dir DIR` (or `save_run`) writes:

- `config.yaml`: the fully resolved config, including the seed; written when training starts (needed by
  `--resume`). Reuse it with `--config DIR/config.yaml` to reproduce the run.
- `metadata.yaml`: timestamp, git commit and dirty flag, Python/torch/gpytorch/CUDA versions, the last
  `epoch` and the final losses; for a resumed run also `resumed` (epoch, timestamp and git commit of each
  resumption).
- `model.pt`: `{'state_dict', 'config', 'metadata'}`.
- `checkpoints/model_epoch_XXXX.pt` (only with `training.checkpoint_every=n`): intermediate checkpoints
  after every n epochs, excluding the last epoch, which is saved as `model.pt`. They use the same format, and
  their metadata records the `epoch` and the loss of that epoch. Load one with
  `load_checkpoint('DIR/checkpoints/model_epoch_0010.pt')`.
- `tensorboard/` (unless `training.tensorboard=false`): TensorBoard event file, see below.
- `checkpoints/last.pt` (only while training): the full training state after the last completed epoch, used by
  `--resume`. It is deleted when the run finishes.

### Resuming an interrupted run

The CLI writes `checkpoints/last.pt` after every epoch, replacing the previous one. It holds the weights, Adam
state, learning-rate scheduler, epoch, losses, metadata and the random number states (python `random`, torch
CPU, CUDA, MPS), so a resumed run draws the same batches and context sizes as an uninterrupted one. The file is
written to a temporary file and then renamed, so an interruption while writing keeps the previous state.

```bash
python -m src.train --resume outputs/run5
```

`--resume` reads `DIR/config.yaml` and continues after the epoch stored in `last.pt`. The config cannot be
changed (e.g. more epochs would change the learning-rate schedule). TensorBoard continues in the same folder; the
values of epochs after `last.pt` (logged before the interruption) are discarded, so every curve is continuous.
On CPU and MPS the result was verified to be identical (bitwise) to an uninterrupted run. On CUDA, kernels that
are not deterministic can cause small differences. At most the epoch that was running is lost.

### Monitoring with TensorBoard

The command line (`TensorBoardLogger` in `train.py`) writes after every epoch, so a running training can be
watched live:

- **Scalars**: `loss/mean` (mean loss of the epoch), `loss_by_context_size/XXXX` (loss per number of context
  points; context sizes not sampled in an epoch are skipped), `optim/lr`, and `time/*` (epoch, data, step and
  forward time).
- **Scalars, with `training.gp_benchmark`** (in the evaluated epochs): `gp_benchmark/<metric>` (all metrics of
  the exact-GP comparison, averaged over subsets and context sizes), `gp_benchmark_kl/<subset>/<task>` and
  `gp_benchmark_nll_gap/<subset>/<task>` (per subset), and `gp_benchmark_kl_by_context_size/XXXX` (averaged over
  subsets).
- **Text**: the resolved config (with the seed) and the run metadata as YAML.
- **HParams**: every config value as a flat key (e.g. `prior.hyperparameters.variance.high`), with the latest
  `loss/mean` as metric (and `gp_benchmark/kl`, `gp_benchmark/nll_gap` with `training.gp_benchmark`). Use it to compare runs by their settings; running or interrupted runs appear too.

```bash
tensorboard --logdir outputs                    # all runs; open http://localhost:6006
tensorboard --logdir outputs/run5/tensorboard   # a single run
```

From Python, pass `epoch_fn=logger.log_epoch` to `train` and call `logger.close()` at the end.

### Using a trained model

```python
import torch
from src.train import load_checkpoint

model, config, metadata = load_checkpoint('outputs/my_run')    # model in eval mode, on CPU

x_train = torch.rand(10, 1, 1)                                  # (n_train, batch_size=1, num_features)
y_train = torch.sin(6 * x_train[..., 0])                        # (n_train, 1)
x_test = torch.linspace(0, 1, 50)[:, None, None]                # (n_test, 1, num_features)

x = torch.cat([x_train, x_test])
y = torch.cat([y_train, torch.zeros(len(x_test), 1)])           # query y values are ignored
with torch.no_grad():
    mean, var = model((x, y), single_eval_pos=len(x_train))     # each (n_test, 1)
```

Keep inputs in the range the model was trained on (`x` in [0, 1], y on the scale of the prior) and the
number of context points below `bptt`. The model has only seen such datasets.

#### Datasets of different sizes in one batch (`inference.py`)

```python
from src.inference import pad_datasets, predict

# contexts: list of (x (n_i, num_features), y (n_i,)); queries: list of x (m_i, num_features)
predictions = predict(model, contexts, queries, batch_size=256)  # list of (mean, var), each (m_i,)

batch = pad_datasets(contexts, queries)   # x, y, single_eval_pos, src_key_padding_mask, query_mask
mean, var = model((batch['x'], batch['y']), single_eval_pos=batch['single_eval_pos'],
                  src_key_padding_mask=batch['src_key_padding_mask'])  # valid where batch['query_mask']
```

Context points are left-padded and queries right-padded. `python -m src.inference --checkpoint DIR` runs a
demo: it samples series of different lengths from the run's prior, forecasts the last points and checks
that the padded batch matches predicting one series at a time (about 1e-6). It also shows how much worse
the predictions are without the mask, because padded `(0, 0)` tokens then act as observations.

### Comparison with the exact GP (`gp_benchmark.py`)

A PFN should approximate the GP posterior predictive. `benchmarks/gp_benchmark_v1.npz` (about 2 MB, kept in git)
holds 16 subsets: 8 fixed hyperparameter settings (`SUBSETS`: short/long RBF lengthscale, low/high noise,
linear trend, trend + RBF, periodic, trend + periodic), each with an `interpolation` task (x uniform, random
order) and a `forecast` task (x sorted, the context consists of the smallest x). Every subset has 256 sequences
of 64 points, sampled exactly (Cholesky, float64, CPU, seed 0).

```bash
python -m src.gp_benchmark evaluate --checkpoint outputs/my_run --output outputs/my_run/gp_benchmark.yaml
python -m src.gp_benchmark create   # only to rebuild the file (bump VERSION when SUBSETS change)
```

For every context size n (0, 1, 2, 4, 8, 16, 32, 48) the first n points are the context and the rest the
queries. The exact posterior predictive is computed in float64 and compared with the PFN per query point:
`kl` = KL(exact GP || PFN) (main metric), `nll_pfn`, `nll_gp`, `nll_gap = nll_pfn - nll_gp`, `mean_z` (RMS
error of the mean in posterior standard deviations) and `log_var_ratio` (> 0: PFN too uncertain). The script
prints the metrics averaged over context sizes (plus an `all subsets` row) and the KL per context size.

The data are stored instead of regenerated from a seed, because random numbers differ between devices and
torch versions, and later changes to `prior.py` must not change the benchmark. The exact GP knows each
subset's true hyperparameters, so it is a best case: a PFN trained on a hyperprior averages over hyperparameters
and is wider, especially for small contexts. `kl` and `nll_gap` are therefore > 0 even for a perfect PFN and are
meant for comparing models (they go to 0 only for a model trained on the subset's fixed hyperparameters). All
`SUBSETS` lie inside the ranges of `configs/gp_hyperprior.yaml`.

During training, `training.gp_benchmark=benchmarks/gp_benchmark_v1.npz` evaluates the benchmark every
`training.gp_benchmark_every` epochs and after the last epoch. The results are printed and logged to TensorBoard
(see "Monitoring with TensorBoard"). `configs/gp_hyperprior.yaml` enables this every 5 epochs; one evaluation
takes a few seconds (about 3 s for a small model on MPS). In Docker, mount `benchmarks/`.

### Evaluation on M4 (`download_m4.py`, `evaluate_m4.py`)

The M4 data are not in git. Download them once (standard library only, no extra dependencies):

```bash
python -m src.download_m4                        # all 6 frequencies + M4-info.csv (about 275 MB)
python -m src.download_m4 --frequencies Yearly   # only what the Yearly evaluation needs (about 31 MB)
```

The files go to `datasets/m4/Train/<Frequency>-train.csv`, `datasets/m4/Test/<Frequency>-test.csv` and
`datasets/m4/M4-info.csv` (`--data_dir` to change the folder). Existing files are skipped (`--force` downloads
them again), and each file is written under a temporary name and renamed when complete, so an interrupted
download can simply be restarted. Then evaluate a run:

```bash
python -m src.evaluate_m4 --checkpoint outputs/my_run --frequency Yearly
python -m src.evaluate_m4 --checkpoint outputs/my_run --frequency Monthly --max_series 1000 \
    --save_predictions outputs/my_run/m4_monthly.csv
```

It prints sMAPE, MASE and OWA of the PFN and of the naive, seasonal naive and drift baselines, plus the
coverage and MSIS of the PFN's 95% interval. `--data_dir` must point to the same folder as for the download.
In Docker, mount `datasets/` (e.g. `-v "$PWD/datasets:/app/datasets"`).

## Docker

The `Dockerfile` in the repository root runs `python -m src.train` as its entry point:

```bash
docker buildx build --platform linux/amd64 -t pfn-train .
docker run --rm --gpus all \
    -v "$PWD/configs:/app/configs:ro" -v "$PWD/benchmarks:/app/benchmarks:ro" -v "$PWD/outputs:/app/outputs" \
    pfn-train --config configs/gp_hyperprior.yaml training.epochs=50 --output_dir outputs/run1
# after an interruption (same mounts):
docker run --rm --gpus all \
    -v "$PWD/configs:/app/configs:ro" -v "$PWD/benchmarks:/app/benchmarks:ro" -v "$PWD/outputs:/app/outputs" \
    pfn-train --resume outputs/run1
```

The image contains only the code and its dependencies. `configs/` and `benchmarks/` are mounted from the host
(read-only), so edited configs need no rebuild; they must exist on the machine that runs the container (e.g. a
repository checkout). A missing host folder is mounted as an empty folder, so the container sees no configs.
Inside the container `git_commit` in `metadata.yaml` is `null`, because the image contains no `.git` folder.

## Origin, license and citation

The code started as a minimal, refactored subset of
[automl/TransformersCanDoBayesianInference](https://github.com/automl/TransformersCanDoBayesianInference),
which is licensed under the Apache License 2.0. This project is distributed under the same license; the full text
is in `src/LICENSE`.

| File | Origin |
|---|---|
| `transformer.py` | Copied from the original `transformer.py` and modified (positional encodings removed, Gaussian head). |
| `train.py` | Refactored from the original `train.py` (configs, run folders, checkpoints, resume, TensorBoard). |
| `utils.py` | Parts copied and simplified from the original `utils.py` (LR schedule, `SeqBN`, `get_batch_to_dataloader`). |
| `prior.py` | New, but follows the `get_batch` interface of the original `priors/fast_gp.py`. |
| `config.py`, `inference.py`, `gp_benchmark.py`, `evaluate_m4.py`, `download_m4.py` | New. |

Apache-2.0 obligations when redistributing: keep the license text (`LICENSE`), keep the attribution notes
(this section and the module docstrings of the derived files) and state that the derived files were modified.
The original repository has no NOTICE file, so there is no NOTICE to carry over.

The M4 data downloaded by `download_m4.py` are not part of this project; they come from
[Mcompetitions/M4-methods](https://github.com/Mcompetitions/M4-methods) and keep that repository's terms.

If you use this code, please cite the original paper:

```bibtex
@inproceedings{muller2022transformers,
  title     = {Transformers Can Do Bayesian Inference},
  author    = {M{\"u}ller, Samuel and Hollmann, Noah and Pineda Arango, Sebastian and Grabocka, Josif and Hutter, Frank},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2112.10510}
}
```
