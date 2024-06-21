import torch
from .base import RMSNorm

_RMSNORM_MODULES = {"torch": RMSNorm}


def get_rmsnorm(
    normalized_shape: int,
    eps: float,
    normalization_implementation: str = "torch",
) -> torch.nn.LayerNorm:
    if normalization_implementation in _RMSNORM_MODULES:
        return _RMSNORM_MODULES[normalization_implementation](
            normalized_shape=normalized_shape, eps=eps
        )

    raise ValueError(
        f"unexpected `normalization_implementation` {normalization_implementation}"
    )
