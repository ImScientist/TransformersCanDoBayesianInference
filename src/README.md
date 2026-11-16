# `src/`: training a Prior-Fitted Network (PFN) on a Gaussian-process prior

This package trains a transformer to do **approximate Bayesian inference for regression in a single forward
pass**. The approach comes from *Transformers Can Do Bayesian Inference* (Müller et al., ICLR 2022).

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
| `utils.py` | Device selection, learning-rate schedule, `single_eval_pos` samplers, `SeqBN`, and `get_batch_to_dataloader` (turns a `get_batch` function into a data loader). |
| `positional_encodings.py` | `NoPositionalEncoding`: the identity. The PFN must not depend on the order of the context points, so no positional information is added. |

## Setup

All commands are run from the **repository root** (the folder that contains `src/`, `configs/`,
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
    training.batch_size=8 model.emsize=64 model.nlayers=1 prior.device=cpu training.device=cpu
```

Command-line arguments:

- `--config FILE`: a YAML file. It can be given several times; files are merged in order.
- `key.sub=value`: overrides a single value (parsed as YAML), e.g. `model.emsize=256`,
  `optim.lr=null`, or `prior.hyperparameters.noise={dist: uniform, low: 0.01, high: 0.1}`.
- `--output_dir DIR`: the run folder. It must not already contain a `model.pt`; this is checked before
  training starts.
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

The names `outputscale` and `lengthscale` (used before the rename to `rbf_*`) are still accepted with a
`DeprecationWarning`, so older configs and checkpoints keep working.

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
  kernel parts, including the periodic part. A good template for new experiments.

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
  prediction therefore depends only on the context and its own `x`, and the context order does not
  matter.
- An MLP decoder maps each query token to two numbers. `GaussianTransformerModel` turns them into
  `mean` and `var = softplus(raw) + min_var`. Its loss is `model.criterion = nn.GaussianNLLLoss(full=True)`.
- `forward((x, y), single_eval_pos)` returns `(mean, var)`, each of shape
  `(seq_len - single_eval_pos, batch_size)`. The `y` values at query positions are ignored.

## Training (`train.py`)

`train(config)` builds the model and the prior data loader, then runs Adam with gradient clipping (norm 1)
and the warm-up + cosine learning-rate schedule. For every batch, a new `single_eval_pos` (the number of
context points) is drawn, so the model learns to predict from any context size between 0 and `bptt - 1`.
After each epoch it prints the mean loss and the loss per `single_eval_pos` (`nan` = position not sampled
in that epoch). It returns `(mean loss of last epoch, losses per position, model on CPU)`.

```python
from src.config import load_config
from src.train import train, save_run, run_metadata

cfg = load_config('configs/gp_hyperprior.yaml', overrides=['training.epochs=5'])
metadata = run_metadata()
loss, pos_losses, model = train(cfg)
save_run('outputs/notebook_run', cfg, model, loss, pos_losses, metadata)
```

### Run folders and checkpoints

`--output_dir DIR` (or `save_run`) writes:

- `config.yaml`: the fully resolved config, including the seed. Reuse it with `--config DIR/config.yaml`
  to reproduce the run.
- `metadata.yaml`: timestamp, git commit and dirty flag, Python/torch/gpytorch/CUDA versions, final losses.
- `model.pt`: `{'state_dict', 'config', 'metadata'}`.

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

## Docker

The `Dockerfile` in the repository root runs `python -m src.train` as its entry point:

```bash
docker buildx build --platform linux/amd64 -t pfn-train .
docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" pfn-train \
    --config configs/gp_hyperprior.yaml training.epochs=50 --output_dir outputs/run1
```

`configs/` is part of the image. Inside the container `git_commit` in `metadata.yaml` is `null`, because
the image contains no `.git` folder.
