"""
Various utilities for neural networks (JAX/Flax version).
"""

import math
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import flax.linen as nn

Array = jnp.ndarray
PRNGKey = jax.Array


# ---------------------------------------------------------------------------
# Activations / Normalization
# ---------------------------------------------------------------------------

class SiLU(nn.Module):
    """SiLU activation as a Flax module (giống class SiLU trong PyTorch version)."""
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Flax đã có nn.silu, nhưng để giống original ta viết tay
        return x * nn.sigmoid(x)


class GroupNorm32(nn.Module):
    """
    GroupNorm nhưng luôn tính trên float32 rồi cast lại dtype gốc,
    tương tự PyTorch GroupNorm32.
    """
    num_groups: int
    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x32 = x.astype(jnp.float32)
        y32 = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
        )(x32)
        return y32.astype(x.dtype)


def normalization(channels: int) -> nn.Module:
    """
    Make a standard normalization layer.

    PyTorch bản gốc: GroupNorm32(32, channels)
    Ở Flax, GroupNorm không cần channels, nên ta chỉ dùng num_groups=32.
    """
    return GroupNorm32(num_groups=32)


# ---------------------------------------------------------------------------
# Layers: conv / linear / pooling
# ---------------------------------------------------------------------------

class ConvND(nn.Module):
    """
    Generic N-D convolution wrapper cho nn.Conv.

    Flax nn.Conv tự động xử lý chiều không gian dựa trên kernel_size.
    Thay vì có Conv1d/2d/3d riêng, ta dùng một lớp chung.
    """
    features: int
    kernel_size: Iterable[int]
    strides: Iterable[int] | None = None
    padding: str = "SAME"
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return nn.Conv(
            features=self.features,
            kernel_size=tuple(self.kernel_size),
            strides=None if self.strides is None else tuple(self.strides),
            padding=self.padding,
            use_bias=self.use_bias,
        )(x)


def conv_nd(dims: int, in_channels: int, out_channels: int,
            kernel_size: int | Iterable[int], **kwargs) -> nn.Module:
    """
    Create a 1D, 2D, or 3D convolution module.

    PyTorch:
        conv_nd(2, in_ch, out_ch, 3, padding=1) -> nn.Conv2d(...)
    JAX/Flax:
        trả về ConvND(...) dùng nn.Conv nội bộ.

    Ghi chú:
    - Flax mặc định dùng layout NHWC. Nếu bạn đang dùng NCHW, cần permute khi gọi.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * dims
    strides = kwargs.pop("stride", None)
    padding = kwargs.pop("padding", "SAME")

    # Các kwargs PyTorch khác (dilation, groups, ...) nếu cần thì thêm vào đây.
    return ConvND(
        features=out_channels,
        kernel_size=kernel_size,
        strides=(strides,) * dims if isinstance(strides, int) else strides,
        padding=padding,
        use_bias=kwargs.get("bias", True),
    )


def linear(in_features: int, out_features: int, bias: bool = True) -> nn.Module:
    """
    Create a linear module (Dense).

    Flax Dense không cần in_features ở constructor, nên tham số in_features chỉ để giữ API giống.
    """
    return nn.Dense(features=out_features, use_bias=bias)


class AvgPoolND(nn.Module):
    """
    Wrapper average pooling N-D dùng nn.avg_pool.
    """
    dims: int
    kernel_size: Iterable[int]
    strides: Iterable[int] | None = None
    padding: str = "VALID"

    @nn.compact
    def __call__(self, x: Array) -> Array:
        window_shape = tuple(self.kernel_size)
        strides = window_shape if self.strides is None else tuple(self.strides)
        return nn.avg_pool(x, window_shape=window_shape, strides=strides, padding=self.padding)


def avg_pool_nd(dims: int, kernel_size: int | Iterable[int],
                stride: int | Iterable[int] | None = None,
                padding: str = "VALID") -> nn.Module:
    """
    Create a 1D, 2D, or 3D average pooling module.

    PyTorch:
        avg_pool_nd(2, 2, 2) -> nn.AvgPool2d(kernel_size=2, stride=2)
    JAX/Flax:
        trả về AvgPoolND, dùng nhƣ một Module.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * dims
    if stride is None:
        strides = None
    elif isinstance(stride, int):
        strides = (stride,) * dims
    else:
        strides = stride
    return AvgPoolND(dims=dims, kernel_size=kernel_size, strides=strides, padding=padding)


# ---------------------------------------------------------------------------
# EMA, scaling, zeroing – trên param pytree
# ---------------------------------------------------------------------------

def update_ema(target_params, source_params, rate: float = 0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    Trong JAX, param là pytree bất biến, nên hàm này
    TRẢ VỀ pytree mới cho target_params.

    :param target_params: pytree params EMA hiện tại.
    :param source_params: pytree params từ model chính.
    :param rate: EMA rate (gần 1 thì update chậm).
    :return: pytree params EMA mới.
    """
    return jax.tree.map(
        lambda targ, src: rate * targ + (1.0 - rate) * src,
        target_params,
        source_params,
    )


def zero_module(params):
    """
    Zero out all parameters in a pytree và trả về pytree mới.

    Ở PyTorch, zero_module(module) chỉnh in-place module.parameters().
    Ở JAX/Flax, ta thao tác trực tiếp trên dict params.
    """
    return jax.tree.map(lambda p: jnp.zeros_like(p), params)


def scale_module(params, scale: float):
    """
    Scale all parameters in a pytree và trả về pytree mới.

    PyTorch version sửa in-place, ở đây trả về param mới.
    """
    return jax.tree.map(lambda p: p * scale, params)


# ---------------------------------------------------------------------------
# Misc utils
# ---------------------------------------------------------------------------

def mean_flat(tensor: Array) -> Array:
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=tuple(range(1, tensor.ndim)))


def timestep_embedding(timesteps: Array, dim: int = 64, max_period: int = 10000) -> Array:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D array of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] array of positional embeddings.
    """
    timesteps = jnp.asarray(timesteps)
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(start=0, stop=half, dtype=jnp.float32)
        / half
    )
    # broadcast: (N, 1) * (1, half) -> (N, half)
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        # pad thêm 1 chiều zero nếu dim lẻ
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])],
            axis=-1,
        )
    return embedding


# ---------------------------------------------------------------------------
# Checkpointing (rematerialization) – gần tương đương
# ---------------------------------------------------------------------------

def checkpoint(func, inputs, params, flag: bool):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    PyTorch version dùng custom autograd.Function với inputs + params.
    Trong JAX/Flax, params thường được đóng trong closure thông qua apply(),
    nên ta dùng jax.checkpoint (remat) lên hàm func.

    :param func: hàm cần evaluate (thường kiểu f(*inputs)).
    :param inputs: tuple/list các input cho func.
    :param params: (không dùng trực tiếp trong JAX version; giữ để tương thích API).
    :param flag: nếu False, không bật gradient checkpointing.
    """
    if flag:
        # jax.checkpoint = jax.remat
        ckpt_func = jax.checkpoint(func)
        return ckpt_func(*inputs)
    else:
        return func(*inputs)

# Public API
__all__ = [
    "SiLU",
    "GroupNorm32",
    "normalization",
    "conv_nd",
    "linear",
    "avg_pool_nd",
    "update_ema",
    "zero_module",
    "scale_module",
    "mean_flat",
    "timestep_embedding",
    "checkpoint",
]
