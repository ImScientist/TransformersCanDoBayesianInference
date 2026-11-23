"""PFN Transformer model.

Copied from `transformer.py` of github.com/automl/TransformersCanDoBayesianInference (Apache-2.0) and modified.
The legacy `fuse_x_y` option was removed: the input is always an
`(x, y)` tuple and `single_eval_pos` is always required. Removing that option also removed
the code paths for `single_eval_pos=None` (causal mask, interleaved x/y tokens). They could
not be reached anyway. There is no positional encoding: the model must not depend on the order of the
context points (all tokens are treated as a set; the attention mask alone separates context and queries).
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .utils import SeqBN


class BaseTransformerModel(nn.Module):
    """PFN Transformer backbone shared by all output heads.

    Encodes the `(x, y)` context and the `x` queries, runs the transformer and decodes the query positions
    into `n_out` raw values each. Subclasses define how these values are turned into a prediction and
    which loss is used:

    - `postprocess(output)`: maps the raw decoder output `(num_queries, batch_size, n_out)` to the
      model prediction returned by `forward`
    - attribute `criterion`: loss module matching that prediction

    :param n_out: number of raw outputs of the decoder per query point
    :param decoder: optional factory `decoder(ninp, nhid, n_out)`; default is an MLP
    """

    def __init__(
            self,
            encoder, ninp, nhead, nhid, nlayers, n_out,
            dropout=0.0,
            y_encoder=None, decoder=None,
            input_normalization=False
    ):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.n_out = n_out
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.decoder = (
            decoder(ninp, nhid, n_out) if decoder is not None
            else nn.Sequential(nn.Linear(ninp, nhid),
                               nn.GELU(),
                               nn.Linear(nhid, n_out))
        )
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.criterion = None  # set by subclasses

        self.init_weights()

    def postprocess(self, output):
        """Map the raw decoder output (num_queries, batch_size, n_out) to the model prediction."""
        raise NotImplementedError

    @staticmethod
    def generate_D_q_matrix(sz, query_size):
        """Attention mask: every position attends to all training (context) positions and to itself;
        query positions do not attend to each other. 0 = allowed, -inf = masked."""
        train_size = sz - query_size
        mask = torch.zeros(sz, sz) == 0
        mask[:, train_size:].zero_()
        mask |= torch.eye(sz) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def add_padding_to_mask(src_mask, src_key_padding_mask, nhead):
        """Combine the shared attention mask with a per-sample padding mask.

        No token attends to a padded token, so every real token's output equals the one of the same dataset
        without padding. Padded tokens may still attend to themselves; this keeps every attention row
        finite (a fully masked row gives NaN, and NaN values leak into other rows even with weight 0).

        :param src_mask: float mask (seq_len, seq_len), 0 = allowed, -inf = masked
        :param src_key_padding_mask: bool (batch_size, seq_len), True = padding
        :param nhead: number of attention heads
        :return: float mask (batch_size * nhead, seq_len, seq_len), the layout expected by `nn.TransformerEncoder`
        """
        batch_size, seq_len = src_key_padding_mask.shape
        mask = src_mask.expand(batch_size, seq_len, seq_len).masked_fill(
            src_key_padding_mask[:, None, :], float('-inf'))
        mask.diagonal(dim1=1, dim2=2).masked_fill_(src_key_padding_mask, 0.)
        return mask.repeat_interleave(nhead, dim=0)

    def init_weights(self):
        # Zero the residual-branch outputs so every encoder layer starts as the identity.
        for layer in self.transformer_encoder.layers:
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            nn.init.zeros_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)

    def forward(self, src, src_mask=None, single_eval_pos=None, src_key_padding_mask=None):
        """
        :param src: tuple `(x, y)` with x: (seq_len, batch_size, num_features), y: (seq_len, batch_size)
        :param src_mask: optional attention mask (seq_len, seq_len); defaults to `generate_D_q_matrix`
        :param single_eval_pos: number of training (context) positions; positions from here on are queries
        :param src_key_padding_mask: optional bool (batch_size, seq_len), True = padded position. Lets datasets
            with different numbers of context / query points share a batch (see `add_padding_to_mask`).
            Padded inputs must be finite (e.g. 0); outputs at padded query positions are meaningless.
            With `input_normalization` use eval mode, otherwise padding enters the batch-norm statistics.
        :return: `self.postprocess(output)` of the decoder output at the query positions,
            output shape: (seq_len - single_eval_pos, batch_size, n_out)
        """
        assert single_eval_pos is not None, 'single_eval_pos is required.'
        assert isinstance(src, tuple), 'src must be a tuple (x, y).'
        x_src, y_src = src

        if src_mask is None:
            src_mask = self.generate_D_q_matrix(len(x_src), len(x_src) - single_eval_pos).to(x_src.device)

        if src_key_padding_mask is not None:
            assert src_key_padding_mask.dtype == torch.bool, 'src_key_padding_mask must be a bool tensor.'
            assert src_key_padding_mask.shape == x_src.shape[1::-1], \
                f'src_key_padding_mask must have shape (batch_size, seq_len) = {tuple(x_src.shape[1::-1])}.'
            nhead = self.transformer_encoder.layers[0].self_attn.num_heads
            src_mask = self.add_padding_to_mask(src_mask, src_key_padding_mask.to(src_mask.device), nhead)

        x_src = self.encoder(x_src)
        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        src = torch.cat([train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output[single_eval_pos:])
        return self.postprocess(output)


class GaussianTransformerModel(BaseTransformerModel):
    """PFN predicting a Gaussian `N(mean, var)` for every query point.

    The decoder produces two raw values per query point: the mean and an unconstrained value that is
    mapped to the variance via `softplus(raw) + min_var` (smooth, positive, ~0.69 at init).

    :param min_var: lower bound added to the predicted variance for numerical stability
    Other parameters: see `BaseTransformerModel` (`n_out` is fixed to 2).

    Attribute `criterion`: `nn.GaussianNLLLoss(reduction='none', full=True)`, called as
    `criterion(mean, target, var=var)`.
    """

    def __init__(self, encoder, ninp, nhead, nhid, nlayers, min_var=1e-6, **kwargs):
        super().__init__(encoder, ninp, nhead, nhid, nlayers, n_out=2, **kwargs)
        self.min_var = min_var
        # Loss matching the predicted Gaussian; per-element losses, incl. the constant term.
        self.criterion = nn.GaussianNLLLoss(reduction='none', full=True)

    def postprocess(self, output):
        """:return: `(mean, var)`, each of shape (num_queries, batch_size)"""
        mean = output[..., 0]
        var = nn.functional.softplus(output[..., 1]) + self.min_var
        return mean, var


class BetaBinomialNLLLoss(nn.Module):
    """Negative log-likelihood of a Beta-Binomial distribution (no reduction).

    `k ~ BetaBinomial(n, alpha, beta)`, i.e. `p ~ Beta(alpha, beta)`, `k ~ Binomial(n, p)`:

        log P(k) = log C(n, k) + lbeta(k + alpha, n - k + beta) - lbeta(alpha, beta)

    with `lbeta(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)`.
    Called as `criterion(alpha, beta, target, total_count)`; all arguments broadcast against each other,
    `target` (k) and `total_count` (n) may be float tensors holding integer values with `0 <= k <= n`.
    """

    @staticmethod
    def _lbeta(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    def forward(self, alpha, beta, target, total_count):
        k = torch.as_tensor(target, dtype=alpha.dtype, device=alpha.device)
        n = torch.as_tensor(total_count, dtype=alpha.dtype, device=alpha.device)
        log_binom = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
        log_prob = log_binom + self._lbeta(k + alpha, n - k + beta) - self._lbeta(alpha, beta)
        return -log_prob


class BetaBinomialTransformerModel(BaseTransformerModel):
    """PFN predicting a Beta-Binomial distribution `BetaBinomial(n, alpha, beta)` for every query point.

    The decoder produces two raw values per query point, mapped to the concentrations via
    `alpha = softplus(raw_0) + min_concentration`, `beta = softplus(raw_1) + min_concentration`.
    The number of trials `n` is not predicted; it is passed to the loss together with the target.

    :param min_concentration: lower bound added to alpha and beta for numerical stability
    Other parameters: see `BaseTransformerModel` (`n_out` is fixed to 2).

    Attribute `criterion`: `BetaBinomialNLLLoss()`, called as `criterion(alpha, beta, target, total_count)`.
    """

    def __init__(self, encoder, ninp, nhead, nhid, nlayers, min_concentration=1e-6, **kwargs):
        super().__init__(encoder, ninp, nhead, nhid, nlayers, n_out=2, **kwargs)
        self.min_concentration = min_concentration
        self.criterion = BetaBinomialNLLLoss()

    def postprocess(self, output):
        """:return: `(alpha, beta)`, each of shape (num_queries, batch_size)"""
        alpha = nn.functional.softplus(output[..., 0]) + self.min_concentration
        beta = nn.functional.softplus(output[..., 1]) + self.min_concentration
        return alpha, beta
