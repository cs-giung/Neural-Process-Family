from ..typing import *

import math
import jax

from jax import numpy as jnp
from jax.scipy import stats
from flax import linen as nn

from .base import NPF
from .. import functional as F
from ..data import NPData
from ..utils import npf_io, MultivariateNormalDiag
from ..modules import (
    # UNet,
    CNN,
    Discretization1d,
    SetConv1dEncoder,
    SetConv1dDecoder,
)


__all__ = [
    # "ConvCNPBase",
    "ConvCNP2d",
]


# TODO: Change model to support on-the-grid(discretized) data.

# NOTE: Currently, `SetConv*` modules to not transforms the output dimension (input_dim = output_dim).
#       This is based on the original definition described in the paper.
#       However, the official implementation transforms the output dimension with pointwise transformation.
#       See (https://github.com/cambridge-mlg/convcnp/blob/3aa2d9c96ff42e55a3c0d8384d084459f19d00f5/convcnp/set_conv.py#L120).
#       In the future, we also need to change the implementation of `SetConv*` modules to match the official implementation.
#       This implementation uses some workaround to mimic some of the behavior of the official implementation.
#           (`mu_log_sigma = nn.Dense(features=(2 * data.y.shape[-1]))(y)`)


class ConvCNPBase(NPF):
    """
    Base class of Convolutional Conditional Neural Process
    """

    discretizer:   Optional[nn.Module] = None
    encoder:       nn.Module = None
    cnn:           nn.Module = None
    decoder:       nn.Module = None
    min_sigma:     float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.encoder is None:
            raise ValueError("encoder is not specified")
        if self.cnn is None:
            raise ValueError("cnn is not specified")

    @nn.compact
    @npf_io
    def __call__(
        self,
        data: NPData,
    ) -> Tuple[Array[B, [T], Y], Array[B, [T], Y]]:

        # image completion
        # data.x_ctx.shape    = [batch, 32, 32, 2]
        # data.mask_ctx.shape = [batch, 32, 32]    -> [batch, 32, 32, 2]
        # data.y_ctx.shape    = [batch, 32, 32, 3]

        # TODO: implement self.encoder
        signal  = self.encoder(data.x_ctx)                                                          # [batch, 32, 32, y_dim]
        density = self.encoder(jnp.repeat(
            jnp.expand_dims(data.mask_ctx, axis=-1), 2, axis=-1).astype(data.x_ctx.dtype))          # [batch, 32, 32, y_dim]
        h = signal / (density + 1e-8)                                                               # [batch, 32, 32, y_dim]
        y = self.cnn(h)
        y = jnp.expand_dims(data.mask, axis=-1) * y                                                 # [batch, 32, 32, 64]

        mu_log_sigma = nn.Dense(features=(2 * data.y.shape[-1]))(y)                                 # [batch, *point, y_dim x 2]
        mu, log_sigma = jnp.split(mu_log_sigma, 2, axis=-1)                                         # [batch, *point, y_dim] x 2
        sigma = self.min_sigma + (1 - self.min_sigma) * nn.softplus(log_sigma)                      # [batch, *point, y_dim]

        mu    = F.masked_fill(mu,    data.mask, fill_value=0., non_mask_axis=-1)                    # [batch, *point, y_dim]
        sigma = F.masked_fill(sigma, data.mask, fill_value=0., non_mask_axis=-1)                    # [batch, *point, y_dim]
        return mu, sigma

    @npf_io
    def log_likelihood(
        self,
        data: NPData,
        *,
        split_set: bool = False,
    ) -> Array:

        mu, sigma = self(data, skip_io=True)                                                        # [batch, *point, y_dim] x 2

        log_prob = MultivariateNormalDiag(mu, sigma).log_prob(data.y)                               # [batch, *point]
        axis = [-i for i in range(1, log_prob.ndim)]

        ll = F.masked_mean(log_prob, data.mask, axis=axis)                                          # [batch]
        ll = jnp.mean(ll)                                                                           # (1)

        if split_set:
            ll_ctx = F.masked_mean(log_prob, data.mask_ctx, axis=axis)                              # [batch]
            ll_tar = F.masked_mean(log_prob, data.mask_tar, axis=axis)                              # [batch]
            ll_ctx = jnp.mean(ll_ctx)                                                               # (1)
            ll_tar = jnp.mean(ll_tar)                                                               # (1)

            return ll, ll_ctx, ll_tar                                                               # (1) x 3
        else:
            return ll                                                                               # (1)

    @npf_io
    def loss(
        self,
        data: NPData,
    ) -> Array:

        loss = -self.log_likelihood(data, skip_io=True)                                             # (1)
        return loss


class AbsConv2d(nn.Module):
    channels: int
    kernel_size: int = 11
    stride: int = 1
    padding: Union[str, int] = 'SAME'
    num_groups: int = 1
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x (Array): An input array with shape [N, H1, W1, C1,].
        
        Returns:
            y (Array): An output array with shape [N, H2, W2, C2,].
        """
        in_channels = x.shape[-1]
        w_shape = (self.kernel_size, self.kernel_size, in_channels // self.num_groups, self.channels,)
        padding = self.padding if isinstance(self.padding, str) else [
            (self.padding, self.padding,), (self.padding, self.padding,),
        ]
        w = self.param('w', jax.nn.initializers.lecun_normal(), w_shape)
        w = jnp.abs(w)
        y = jax.lax.conv_general_dilated(
            x, w, (self.stride, self.stride,), padding,
            lhs_dilation=(1,1,), rhs_dilation=(1,1,),
            dimension_numbers   = jax.lax.ConvDimensionNumbers((0,3,1,2,), (3,2,0,1,), (0,3,1,2,)),
            feature_group_count = self.num_groups,
        )
        return y


# TODO: Add 2d model
class ConvCNP2d:
    """
    Convolutional Conditional Neural Process
    """

    def __new__(cls,
        y_dim: int,
        x_min: float,
        x_max: float,
        r_dim: int = 64,
        cnn_dims: Optional[Sequence[int]] = None,
        cnn_xl: bool = False,
        on_the_grid: bool = False,
        points_per_unit: int = 64,
        x_margin: float = 0.1,
        min_sigma: float = 0.1,
    ):

        # assert on_the_grid is False, "on_the_grid is not supported yet"

        if cnn_xl:
            raise NotImplementedError("cnn_xl is not supported yet")
            Net = UNet
            cnn_dims = cnn_dims or (8, 16, 16, 32, 32, 64)
            multiple = 2 ** len(cnn_dims)  # num_halving_layers = len(cnn_dims)
        else:
            Net = CNN
            cnn_dims = cnn_dims or (r_dim,) * 4
            multiple = 1

        init_log_scale = math.log(2. / points_per_unit)

        if on_the_grid:
            discretizer = None
            encoder = AbsConv2d(channels=y_dim)
            decoder = None
        else:
            discretizer = Discretization1d(
                minval=x_min,
                maxval=x_max,
                points_per_unit=points_per_unit,
                multiple=multiple,
                margin=x_margin,
            )
            encoder = SetConv1dEncoder(init_log_scale=init_log_scale)   # TODO: Support dimension transformation
            decoder = SetConv1dDecoder(init_log_scale=init_log_scale)   # TODO: Support dimension transformation

        cnn = Net(dimension=2, hidden_features=cnn_dims, out_features=r_dim)

        return ConvCNPBase(
            discretizer=discretizer,
            encoder=encoder,
            cnn=cnn,
            decoder=decoder,
            min_sigma=min_sigma,
        )
