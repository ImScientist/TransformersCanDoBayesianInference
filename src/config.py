"""Training configuration: typed defaults, YAML loading/saving and command-line overrides.

The pydantic models below are the single source of truth for all default values and constraints. A
configuration is built by merging (with OmegaConf), in this order (later wins):
    1. YAML files (only the values that differ from the defaults are needed), e.g. `configs/linear_gp.yaml`
    2. dotted overrides, e.g. `["model.emsize=256", "prior.hyperparameters.noise=0.01"]`
and then validated by pydantic, which fills in the defaults of `TrainConfig` for missing values: unknown keys, values of the wrong type and violated constraints raise a
`pydantic.ValidationError` (a `ValueError`) that lists every problem with its path.

Usage in a notebook:
    from src.config import load_config
    cfg = load_config('configs/linear_gp.yaml', overrides=['training.epochs=5'])
    cfg.model.emsize = 256  # assignments are validated too (`validate_assignment`)
Constraints between sections (e.g. `training.steps_per_epoch % optim.aggregate_k_gradients`) are only checked
when the whole config is validated: by `load_config`, `config_from_dict` and at the start of `src.train.train`.
"""
from typing import Annotated, Literal, Self

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, ValidationError, model_validator

PositiveInt = Annotated[int, Field(ge=1)]


class _Config(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    def __setattr__(self, name, value):
        # pydantic assigns before running model validators; restore the old value if they fail.
        old = self.__dict__.get(name)
        try:
            super().__setattr__(name, value)
        except ValidationError:
            self.__dict__[name] = old
            raise


class Distribution(_Config):
    """Distribution of a prior hyperparameter, sampled independently for every dataset (sequence) of a batch
    (see `src.prior.sample_hyperparameters`). `log_uniform` is uniform in log space (requires `low > 0`).
    With `zero_prob > 0` the distribution is zero-inflated: the value is exactly 0 with probability
    `zero_prob`, otherwise drawn from `dist`. Useful for scales that switch kernel parts off for some datasets
    (`variance`, `bias`, `rbf_outputscale`, `periodic_outputscale`); hyperparameters that must be > 0
    (e.g. `noise`, lengthscales, `period`) reject it.

    YAML: `noise: {dist: log_uniform, low: 1.e-4, high: 0.1}`,
          `periodic_outputscale: {dist: log_uniform, low: 0.01, high: 1.0, zero_prob: 0.5}`
    """
    dist: Literal['uniform', 'log_uniform']
    low: float
    high: float
    zero_prob: Annotated[float, Field(ge=0, le=1)] = 0.

    @model_validator(mode='after')
    def _check(self) -> Self:
        if self.low > self.high:
            raise ValueError(f'low must be <= high, got low={self.low}, high={self.high}')
        if self.dist == 'log_uniform' and self.low <= 0:
            raise ValueError('log_uniform requires low > 0')
        return self


# Mappings are validated only as `Distribution`, everything else only as a number (clearer error messages).
Hyperparameter = Annotated[
    Annotated[float, Tag('fixed')] | Annotated[Distribution, Tag('dist')],
    Discriminator(lambda v: 'dist' if isinstance(v, (dict, Distribution)) else 'fixed')]


def _bounds(value):
    """(min, max) of a fixed value or a `Distribution` (including the 0 of a zero-inflated one)."""
    if not isinstance(value, Distribution):
        return value, value
    if value.zero_prob == 1:
        return 0., 0.
    if value.zero_prob > 0:
        return min(value.low, 0.), max(value.high, 0.)
    return value.low, value.high


class GPHyperparameters(_Config):
    """Hyperparameters of the GP prior (see `src.prior.get_batch`). Each is a fixed number or a `Distribution`."""
    noise: Hyperparameter = .1  # observation noise variance; must be > 0
    variance: Hyperparameter = .1  # variance of the linear (dot-product) part
    rbf_outputscale: Hyperparameter = .1  # RBF output scale; 0 = no RBF part
    rbf_lengthscale: Hyperparameter | None = .1  # RBF lengthscale; required if rbf_outputscale can be > 0
    bias: Hyperparameter = 0.  # constant kernel offset (intercept variance); 0 = none
    periodic_outputscale: Hyperparameter = 0.  # Exp-Sine-Squared output scale; 0 = no periodic part
    period: Hyperparameter | None = None  # period; required if periodic_outputscale can be > 0
    periodic_lengthscale: Hyperparameter | None = None  # smoothness within a period; small = more harmonics
    periodic_decay: Hyperparameter | None = None  # lengthscale of the RBF factor; None = exactly periodic

    @model_validator(mode='after')
    def _check(self) -> Self:
        def low(name):
            return _bounds(getattr(self, name))[0]

        def positive_if_needed(scale, names):
            if _bounds(getattr(self, scale))[1] > 0:
                for name in names:
                    if getattr(self, name) is None or low(name) <= 0:
                        errors.append(f'{name} must be > 0 (not null) if {scale} can be > 0')

        errors = []
        if low('noise') <= 0:
            errors.append('noise must be > 0')
        errors += [f'{name} must be >= 0' for name in ('variance', 'rbf_outputscale', 'bias', 'periodic_outputscale')
                   if low(name) < 0]
        positive_if_needed('rbf_outputscale', ['rbf_lengthscale'])
        positive_if_needed('periodic_outputscale', ['period', 'periodic_lengthscale'])
        if self.periodic_decay is not None and low('periodic_decay') <= 0:
            errors.append('periodic_decay must be > 0 or null')
        if errors:
            raise ValueError('; '.join(errors))
        return self


class PriorConfig(_Config):
    """Arguments of the prior DataLoader / `src.prior.get_batch`."""
    num_features: PositiveInt = 1
    equidistant_x: bool = False
    device: str | None = None  # device for sampling the prior; None = `default_device` (use 'cpu' if MPS is unstable)
    # 'auto' = 'basis' if num_features == 1 else 'cholesky'; see `src.prior.get_batch`
    sampling_method: Literal['auto', 'basis', 'cholesky'] = 'auto'
    basis_tol: Annotated[float, Field(ge=1e-6, le=0.1)] = 1e-4  # accuracy of the 'basis' method (float32 limit ~1e-6)
    # fraction of datasets with x sorted (context = smallest x, queries = largest x: forecasting); num_features == 1
    forecast_prob: Annotated[float, Field(ge=0, le=1)] = 0.
    hyperparameters: GPHyperparameters = Field(default_factory=GPHyperparameters)

    @model_validator(mode='after')
    def _check(self) -> Self:
        if self.equidistant_x and self.num_features != 1:
            raise ValueError('equidistant_x requires num_features == 1')
        if self.sampling_method == 'basis' and self.num_features != 1:
            raise ValueError("sampling_method='basis' requires num_features == 1")
        if self.forecast_prob > 0 and self.num_features != 1:
            raise ValueError('forecast_prob > 0 requires num_features == 1')
        return self


class ModelConfig(_Config):
    """Arguments of `src.transformer.GaussianTransformerModel`."""
    emsize: PositiveInt = 512
    nhid: PositiveInt | None = None  # None = 2 * emsize
    nlayers: PositiveInt = 6
    nhead: PositiveInt = 4  # nhead = emsize / 64 in the original paper
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.
    input_normalization: bool = False
    min_var: Annotated[float, Field(ge=0)] = 1e-6

    @model_validator(mode='after')
    def _check(self) -> Self:
        if self.emsize % self.nhead:
            raise ValueError(f'emsize ({self.emsize}) must be divisible by nhead ({self.nhead})')
        return self


class OptimConfig(_Config):
    lr: Annotated[float, Field(gt=0)] | None = .001  # None = `get_openai_lr(model)`
    warmup_epochs: Annotated[int, Field(ge=0)] = 50
    aggregate_k_gradients: PositiveInt = 1


class TrainingConfig(_Config):
    epochs: PositiveInt = 200
    steps_per_epoch: PositiveInt = 10
    batch_size: PositiveInt = 1000
    bptt: Annotated[int, Field(ge=2)] = 10  # sequence length (training + query points) of each sampled dataset
    max_eval_pos: PositiveInt | None = None  # single_eval_pos is sampled from range(max_eval_pos); None = bptt
    eval_pos_sampling: Literal['weighted', 'uniform'] = 'weighted'
    seed: int | None = None  # None = random seed, chosen and recorded by `train`
    device: str | None = None  # device for the model; None = `default_device` (CUDA, MPS, then CPU)
    # Save an intermediate checkpoint every n epochs; None = epochs, i.e. only the final model
    checkpoint_every: PositiveInt | None = None
    tensorboard: bool = True  # write TensorBoard logs to <output_dir>/tensorboard
    # Exact-GP benchmark file (`src/gp_benchmark.py`) evaluated during training; None = no evaluation
    gp_benchmark: str | None = None
    # Evaluate the benchmark every n epochs and after the last one; None = epochs, i.e. only after the last epoch
    gp_benchmark_every: PositiveInt | None = None

    @model_validator(mode='after')
    def _check(self) -> Self:
        if self.max_eval_pos is not None and self.max_eval_pos > self.bptt:
            raise ValueError(f'max_eval_pos must be <= bptt={self.bptt}, got {self.max_eval_pos}')
        return self


class TrainConfig(_Config):
    prior: PriorConfig = Field(default_factory=PriorConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode='after')
    def _check(self) -> Self:
        if self.training.steps_per_epoch % self.optim.aggregate_k_gradients:
            raise ValueError('training.steps_per_epoch must be divisible by optim.aggregate_k_gradients')
        return self


def config_to_dict(config):
    return config.model_dump()


def config_from_dict(data):
    """Build and validate a `TrainConfig` from a nested dict; missing keys keep their defaults."""
    return TrainConfig.model_validate(data)


def load_config(*paths, overrides=()):
    """YAML files (in order) <- dotted overrides `key.sub=value` (values parsed as YAML); missing values take
    the `TrainConfig` defaults."""
    merged = OmegaConf.merge(
        *(OmegaConf.load(path) for path in paths),
        OmegaConf.from_dotlist(list(overrides)))
    return config_from_dict(OmegaConf.to_container(merged, resolve=True))


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.safe_dump(config_to_dict(config), f, sort_keys=False)
