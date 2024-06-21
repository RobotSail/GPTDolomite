from .activations import get_activation_function, is_glu
from .attention import (
    SDPA,
    Attention,
    FlashAttention2,
    PaddingFreeAttention,
    get_attention_module,
    get_unpad_data,
    interleave_query_key_value_tensor_for_attention,
    repeat_key_value,
    split_query_key_value_tensor_for_attention,
)
# from .embedding import ParameterizedEmbedding
# from .linear import ParameterizedLinear

from functools import partial

# from torch.nn import Embedding as ParameterizedEmbedding
# from torch.nn import Linear as ParameterizedLinear
from torch.nn import Embedding
from torch.nn import Linear 

class ParameterizedEmbedding(Embedding):

    def __init__(self, *args, **kwargs):
        kwargs.pop('std')
        super().__init__(*args, **kwargs)

class ParameterizedLinear(Linear):

    def __init__(self, *args, **kwargs):
        kwargs.pop('std')
        super().__init__(*args, **kwargs)

from .normalization import RMSNorm, get_normalization_function
from .position_embedding import Alibi, RoPE, YaRNScaledRoPE, apply_rotary_pos_emb
