import torch
from torch import nn
from math import *
from . import register_encoder
from transformers import AutoModel

@register_encoder()
class InternViTwNorm(nn.Module):
    def __init__(
        self,
        model_path: str,
        normalize: bool = True,
        downsample_ratio: float = 0.5,
    ):
        super().__init__()
        # InternViT is not natively in transformers; load via AutoModel with trust_remote_code
        try:
            self.encoder = AutoModel.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = AutoModel.from_pretrained(
                model_path, local_files_only=False, trust_remote_code=True
            )
        self.encoder.requires_grad_(False)
        if normalize and hasattr(self.encoder, 'layernorm'):
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.num_heads = self.encoder.config.num_attention_heads

        # Pixel shuffle + MLP for 2x2 token merge (following InternVL)
        self.llm_hidden_size = 2048 if '2B' in model_path else 1024
        self.downsample_ratio = downsample_ratio
        merge_channel_dim = self.hidden_size * int(1 / self.downsample_ratio) ** 2  # hidden_size * 4 for 2x2 merge
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(merge_channel_dim),
            nn.Linear(merge_channel_dim, self.llm_hidden_size),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
        )
        self.merged_hidden_size = self.llm_hidden_size

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        """Merge spatial tokens via pixel-shuffle downsampling.

        Input:  (B, W, H, C)
        Output: (B, W*scale, H*scale, C / scale^2)
        For scale_factor=0.5 this performs 2x2 merge: spatial dims halve, channels x4.
        """
        batch_size, width, height, channels = x.size()
        new_height = int(height * scale_factor)
        new_channels_step1 = int(channels / scale_factor)
        x = x.view(batch_size, width, new_height, new_channels_step1)
        x = x.permute(0, 2, 1, 3).contiguous()
        new_width = int(width * scale_factor)
        final_channels = int(channels / (scale_factor * scale_factor))
        x = x.view(batch_size, new_height, new_width, final_channels)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def internvit_forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True)
        unused_token_num = 1  # 1 CLS token
        image_features = outputs.last_hidden_state[:, unused_token_num:]

        # Pixel shuffle 2x2 merge + MLP projection
        height = width = int(image_features.shape[1] ** 0.5)
        image_features = image_features.reshape(image_features.shape[0], height, width, -1)
        image_features = self.pixel_shuffle(image_features, scale_factor=self.downsample_ratio)
        image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[-1])
        image_features = self.mlp1(image_features)

        return image_features, (None, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.internvit_forward(x)
