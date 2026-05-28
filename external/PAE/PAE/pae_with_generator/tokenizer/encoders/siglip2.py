from torch import nn
import torch
from math import *
from . import register_encoder
from transformers import SiglipModel

@register_encoder()
class SigLIP2wNorm(nn.Module):
    def __init__(
        self,
        model_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = SiglipModel.from_pretrained(model_path, local_files_only=True).vision_model
        except (OSError, ValueError, AttributeError):
            self.encoder = SiglipModel.from_pretrained(model_path, local_files_only=False).vision_model
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.post_layernorm.elementwise_affine = False
            self.encoder.post_layernorm.weight = None
            self.encoder.post_layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.num_heads = self.encoder.config.num_attention_heads

    def siglip2_forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True, interpolate_pos_encoding=True)
        image_features = outputs.last_hidden_state
        return image_features, (None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.siglip2_forward(x)
