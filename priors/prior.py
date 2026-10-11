from torch.utils.data import DataLoader


class PriorDataLoader(DataLoader):
    """Interface for prior data loaders consumed by `train.train`.

    Subclasses do not call `DataLoader.__init__`; they only implement `__len__` and `__iter__`
    (yielding `(data, targets)` batches sampled on the fly). Required interface:
      - `__init__(num_steps, fuse_x_y=False, **get_batch_kwargs)`
      - attributes (class or instance level): `num_features: int`, `num_outputs: int`, `fuse_x_y: bool`
      - optional: `validate(model)` called periodically during training
    See `priors.utils.get_batch_to_dataloader` for the standard implementation.
    """
