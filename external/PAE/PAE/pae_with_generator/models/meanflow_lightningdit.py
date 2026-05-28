"""
B3 / global sample-level MeanFlow wrapper for official PAE LightningDiT.

This module is intentionally non-invasive: it reuses the upstream LightningDiT
submodules, but exposes the MeanFlow training signature

    u_theta(z_t, r, t, y, force_drop_ids=None)

where r and t are sample-level scalar tensors of shape [B].  Conditioning uses
emb(t) + emb(t-r) + emb(y), matching the Phase-0/B3 global MeanFlow baseline
smoke recipe.  The standard PAE LightningDiT and FM transport remain unchanged.
"""

from __future__ import annotations

import os
from typing import List, Optional

# Keep the MeanFlow/JVP path out of Dynamo by default.  The upstream file uses
# @torch.compile on several small modules; forward-mode AD transforms are more
# reliable if we unwrap those compiled callables for this trainer.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import models.lightningdit as lightningdit_module
from models.lightningdit import FinalLayer, LabelEmbedder, LightningDiT, LightningDiTBlock, TimestepEmbedder
from models.swiglu_ffn import SwiGLUFFN


def _plain_modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def disable_compile_for_forward_ad() -> List[str]:
    """Unwrap upstream @torch.compile callables for JVP/FD correctness runs."""
    if os.environ.get("PAE_MEANFLOW_ENABLE_DYNAMO", "0") == "1":
        return []

    unwrapped: List[str] = []
    lightningdit_module.modulate = _plain_modulate
    unwrapped.append("models.lightningdit.modulate->plain_python")

    for cls in (SwiGLUFFN, TimestepEmbedder, LabelEmbedder, LightningDiTBlock, FinalLayer):
        fn = getattr(cls, "forward", None)
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            setattr(cls, "forward", wrapped)
            unwrapped.append(f"{cls.__name__}.forward")
    return unwrapped


COMPILE_UNWRAPPED = disable_compile_for_forward_ad()


def _init_linear_normal(linear: nn.Linear, std: float) -> None:
    nn.init.normal_(linear.weight, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_linear_xavier(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class B3MeanFlowLightningDiT(nn.Module):
    """Official PAE LightningDiT submodules with B3 MeanFlow conditioning.

    Args mirror ``LightningDiT`` with a few MeanFlow-specific additions.  The
    default ``meanflow_init_scheme='official_zero'`` preserves upstream DiT
    zero-output initialization for real training.  Tiny smoke tests may set
    ``stress_nonzero`` to make JVP/FD audits non-trivial in only a few steps.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = False,
        use_qknorm: bool = False,
        use_swiglu: bool = False,
        use_rope: bool = False,
        use_rmsnorm: bool = False,
        wo_shift: bool = False,
        use_checkpoint: bool = False,
        use_abs_pos: bool = True,
        checkpoint_reentrant: bool = False,
        meanflow_init_scheme: str = "official_zero",
        meanflow_adaln_init_std: float = 0.02,
        meanflow_final_linear_init: str = "official_zero",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.class_dropout_prob = class_dropout_prob
        self.meanflow_conditioning = "emb(t)+emb(t-r)+emb(y)"
        self.meanflow_init_scheme = meanflow_init_scheme
        self.use_checkpoint = use_checkpoint
        self.checkpoint_reentrant = checkpoint_reentrant

        # Keep upstream LightningDiT intact as the core, but disable its internal
        # checkpoint flag.  We control checkpointing here and the official mixed
        # recipe rejects checkpoint around the target/JVP path.
        self.core = LightningDiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            use_checkpoint=False,
            use_abs_pos=use_abs_pos,
        )
        self.delta_embedder = TimestepEmbedder(hidden_size)
        self._initialize_meanflow_additions(
            scheme=meanflow_init_scheme,
            adaln_init_std=meanflow_adaln_init_std,
            final_linear_init=meanflow_final_linear_init,
        )

    def _initialize_meanflow_additions(self, scheme: str, adaln_init_std: float, final_linear_init: str) -> None:
        # Match upstream timestep embedding init for the additional delta=t-r embedding.
        _init_linear_normal(self.delta_embedder.mlp[0], 0.02)
        _init_linear_normal(self.delta_embedder.mlp[2], 0.02)

        if scheme == "official_zero":
            # Upstream LightningDiT.initialize_weights has already zeroed block
            # AdaLN and output layers.  This is the intended long-train init.
            return
        if scheme != "stress_nonzero":
            raise ValueError(f"unknown meanflow_init_scheme={scheme!r}")

        # Tiny smoke init: make time dependence and output non-zero so JVP/FD is
        # measurable without a long warmup.
        for block in self.core.blocks:
            _init_linear_normal(block.adaLN_modulation[-1], adaln_init_std)
        _init_linear_normal(self.core.final_layer.adaLN_modulation[-1], adaln_init_std)
        if final_linear_init == "xavier":
            _init_linear_xavier(self.core.final_layer.linear)
        elif final_linear_init == "official_zero":
            nn.init.zeros_(self.core.final_layer.linear.weight)
            if self.core.final_layer.linear.bias is not None:
                nn.init.zeros_(self.core.final_layer.linear.bias)
        else:
            raise ValueError(f"unknown meanflow_final_linear_init={final_linear_init!r}")

    def forward(
        self,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict B3 MeanFlow velocity/average velocity.

        ``r`` and ``t`` must be sample-level tensors of shape [B].  The caller is
        responsible for constructing and reusing ``force_drop_ids`` when class
        dropout is enabled, so the fp32 JVP target and bf16 train forward see
        identical class-conditioning.
        """
        if z.ndim != 4:
            raise ValueError(f"expected z as [B,C,H,W], got {tuple(z.shape)}")
        if r.ndim != 1 or t.ndim != 1 or r.shape != t.shape or r.shape[0] != z.shape[0]:
            raise ValueError(f"expected sample-level r,t [B], got r={tuple(r.shape)} t={tuple(t.shape)} z={tuple(z.shape)}")

        x = self.core.x_embedder(z)
        if self.core.use_abs_pos:
            x = x + self.core.pos_embed.to(device=x.device, dtype=x.dtype)

        delta = t - r
        c = self.core.t_embedder(t) + self.delta_embedder(delta) + self.core.y_embedder(y, self.training, force_drop_ids=force_drop_ids)

        for block in self.core.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, c, self.core.feat_rope, use_reentrant=self.checkpoint_reentrant)
            else:
                x = block(x, c, self.core.feat_rope)

        x = self.core.final_layer(x, c)
        x = self.core.unpatchify(x)
        if self.core.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x


def B3MeanFlowLightningDiT_Tiny_1(**kwargs) -> B3MeanFlowLightningDiT:
    return B3MeanFlowLightningDiT(depth=2, hidden_size=128, patch_size=1, num_heads=4, **kwargs)


def B3MeanFlowLightningDiT_S_1(**kwargs) -> B3MeanFlowLightningDiT:
    return B3MeanFlowLightningDiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)


def B3MeanFlowLightningDiT_B_1(**kwargs) -> B3MeanFlowLightningDiT:
    return B3MeanFlowLightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)


def B3MeanFlowLightningDiT_B_2(**kwargs) -> B3MeanFlowLightningDiT:
    return B3MeanFlowLightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def B3MeanFlowLightningDiT_XL_1(**kwargs) -> B3MeanFlowLightningDiT:
    return B3MeanFlowLightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)


def B3MeanFlowLightningDiT_XL_2(**kwargs) -> B3MeanFlowLightningDiT:
    return B3MeanFlowLightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


B3MeanFlowLightningDiT_models = {
    "B3MeanFlowLightningDiT-Tiny/1": B3MeanFlowLightningDiT_Tiny_1,
    "B3MeanFlowLightningDiT-S/1": B3MeanFlowLightningDiT_S_1,
    "B3MeanFlowLightningDiT-B/1": B3MeanFlowLightningDiT_B_1,
    "B3MeanFlowLightningDiT-B/2": B3MeanFlowLightningDiT_B_2,
    "B3MeanFlowLightningDiT-XL/1": B3MeanFlowLightningDiT_XL_1,
    "B3MeanFlowLightningDiT-XL/2": B3MeanFlowLightningDiT_XL_2,
}
