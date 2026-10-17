import time

import torch
from torch import nn
import gpytorch

from .utils import get_batch_to_dataloader
from utils import default_device


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP with constant mean, scaled RBF kernel and Gaussian likelihood.

    All hyperparameters are fixed at construction time through the `initial_value` of each
    parameter's constraint (gpytorch applies it when the constraint is registered), so no
    attributes need to be edited after initialization.

    :param train_x: training inputs, shape (batch, n, num_features), or None for a prior-only model
    :param train_y: training targets, shape (batch, n), or None for a prior-only model.
        A prior-only model must be put in eval mode (`model.eval()`) before calling it.
    :param noise: observation noise variance of the Gaussian likelihood
    :param outputscale: output scale (signal variance) of the ScaleKernel
    :param lengthscale: lengthscale of the RBF kernel (shared across features)
    :param min_noise: lower bound of the noise constraint
    """

    def __init__(self, train_x, train_y, noise, outputscale, lengthscale, min_noise=1.e-9):
        """ ... """

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(min_noise, initial_value=noise))

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=gpytorch.kernels.RBFKernel(
                lengthscale_constraint=gpytorch.constraints.Positive(initial_value=lengthscale)),
            outputscale_constraint=gpytorch.constraints.Positive(initial_value=outputscale))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_model(x, y, hyperparameters):
    """ Get model and likelihood """

    model = ExactGPModel(
        train_x=x,
        train_y=y,
        noise=hyperparameters["noise"],
        outputscale=hyperparameters["outputscale"],
        lengthscale=hyperparameters["lengthscale"])

    return model, model.likelihood


def get_model_on_device(x, y, hyperparameters, device):
    model, likelihood = get_model(x, y, hyperparameters)
    model.to(device)
    return model, likelihood


@torch.no_grad()
# WARNING: `default_device` is 'mps' on Apple Silicon. MPS has no float64 support, so GP prior sampling
# can be numerically unstable for very small noise (e.g. noise=1e-4). If you see NotPSD/jitter
# warnings or errors, pass device='cpu' here.
def get_batch(
        batch_size,
        seq_len: int,
        num_features: int,
        device=default_device,
        hyperparameters=None,
        equidistant_x=False
):
    if isinstance(hyperparameters, (tuple, list)):
        hyperparameters = {
            "noise": hyperparameters[0],
            "outputscale": hyperparameters[1],
            "lengthscale": hyperparameters[2]}

    elif hyperparameters is None:
        hyperparameters = {
            "noise": .1,
            "outputscale": .1,
            "lengthscale": .1}

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations', (True, True, True))):

        # -> (batch_size, seq_len, num_feat)
        if equidistant_x:
            assert num_features == 1
            x = torch.linspace(0, 1, seq_len, device=device)[:, None].repeat(batch_size, 1, 1)
        else:
            x = torch.rand(batch_size, seq_len, num_features, device=device)

        # No training data: in eval mode the model returns the GP prior N(m(x), K(x, x)) for any x.
        # (In train mode a model without training data would raise an error.)
        model, likelihood = get_model(None, None, hyperparameters)
        model.to(device)
        model.eval()

        # prior_mode is redundant without training data, but makes the intent explicit.
        with gpytorch.settings.prior_mode(True):
            d = model(x)  # prior of the noise-free function f
            d = likelihood(d)  # adds observation noise -> prior of y
            sample = d.sample().transpose(0, 1)

    # -> (seq_len, batch_size, num_feat), (seq_len, batch_size), (seq_len, batch_size)
    return x.transpose(0, 1), sample, sample  # x.shape = (T,B,H)


# TODO: Reintegrate this code
# num_features_used = num_features_used_sampler()
# prior_outputscale = prior_outputscale_sampler()
# prior_lengthscale = prior_lengthscale_sampler()
#
# x, sample = normalize_data(x), normalize_data(sample)
#
# if is_binary_classification:
#     sample = (sample > torch.median(sample, dim=0)[0]).float()
#
# if normalize_by_used_features:
#     x = normalize_by_used_features_f(x, num_features_used, num_features)
#
# # # if is_binary_classification and order_y:
# # #     x, sample = order_by_y(x, sample)
# #
# # Append empty features if enabled
# x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], num_features - num_features_used), device=device)], -1)

DataLoader = get_batch_to_dataloader(get_batch, default_num_outputs=1)


# `get_model_on_device` is a hook for building the GP at every step t. The default uses fixed
# hyperparameters; `priors/fast_gp_mix.py` swaps in `get_fitted_model` (hyperparameters fitted
# by marginal likelihood on x[:t], y[:t]) via functools.partial. Do not remove.
@torch.no_grad()
def evaluate(
        x, y, y_non_noisy,
        use_mse=False,
        hyperparameters={},
        get_model_on_device=get_model_on_device,
        device=default_device,
        step_size=1,
        start_pos=0
):
    start_time = time.time()
    losses_after_t = [.0] if start_pos == 0 else []
    all_losses_after_t = []

    args = hyperparameters.get('fast_computations', (True, True, True))
    with (
        gpytorch.settings.fast_computations(*args),
        gpytorch.settings.fast_pred_var(False)
    ):
        for t in range(max(start_pos, 1), len(x), step_size):
            # Given the first t points of a dataset, what is the distribution of the
            # next point's y? Doing this for every t gives the exact-GP loss curve
            # over context length. The Transformer is compared against that curve.

            model, likelihood = get_model_on_device(
                x[:t].transpose(0, 1),
                y[:t].transpose(0, 1),
                hyperparameters,
                device)

            model.eval()

            # print([t.shape for t in model.train_inputs])
            # print(x[:t].transpose(0,1).shape, x[t].unsqueeze(1).shape, y[:t].transpose(0,1).shape)
            f = model(x[t].unsqueeze(1))
            l = likelihood(f)
            means = l.mean.squeeze()
            varis = l.covariance_matrix.squeeze()
            # print(l.variance.squeeze(), l.mean.squeeze(), y[t])

            assert len(means.shape) == len(varis.shape) == 1
            assert len(means) == len(varis) == x.shape[1]

            if use_mse:
                c = nn.MSELoss(reduction='none')
                ls = c(means, y[t])
            else:
                ls = -l.log_prob(y[t].unsqueeze(1))

            losses_after_t.append(ls.mean())
            all_losses_after_t.append(ls.flatten())

        return (
            torch.stack(all_losses_after_t).to('cpu'),
            torch.tensor(losses_after_t).to('cpu'),
            time.time() - start_time
        )


if __name__ == '__main__':

    hps = {"noise": .1, "outputscale": .1, "lengthscale": .1}
    hps_eval = {'fast_computations': (True, True, True)}

    for redo_idx in range(1):
        print(
            evaluate(*get_batch(batch_size=1_000,
                                seq_len=10,
                                num_features=10,
                                hyperparameters=hps),
                     use_mse=False,
                     hyperparameters=hps_eval)
        )
