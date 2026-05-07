from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from diffusers import UNet2DConditionModel


@dataclass(frozen=True)
class MDPaperLikeConfig:
    """
    Approximation of the MobileDiffusion diffusion-network design (architecture-side).
    """

    sample_size: int = 64
    in_channels: int = 4
    out_channels: int = 4
    cross_attention_dim: int = 768

    block_out_channels: tuple[int, ...] = (320, 640, 1024)
    layers_per_block: int = 1

    down_block_types: tuple[str, ...] = ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D")
    up_block_types: tuple[str, ...] = ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D")

    attention_head_dim: int = 64


class _NoOpAttention(nn.Module):
    def forward(  # type: ignore[override]
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        return hidden_states


def _disable_self_attention_in_transformer(attn_module: nn.Module) -> None:
    for m in attn_module.modules():
        if hasattr(m, "attn1"):
            try:
                m.attn1 = _NoOpAttention()
            except Exception:
                pass


def create_md_paper_like_unet(cfg: MDPaperLikeConfig = MDPaperLikeConfig()) -> UNet2DConditionModel:
    unet = UNet2DConditionModel(
        sample_size=cfg.sample_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=cfg.block_out_channels,
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
        cross_attention_dim=cfg.cross_attention_dim,
        attention_head_dim=cfg.attention_head_dim,
    )

    # Disable self-attention on 32x32 blocks (down_block index 1, up_block index 1).
    if len(getattr(unet, "down_blocks", [])) >= 2:
        db = unet.down_blocks[1]
        if hasattr(db, "attentions"):
            for attn in db.attentions:
                _disable_self_attention_in_transformer(attn)
    if len(getattr(unet, "up_blocks", [])) >= 2:
        ub = unet.up_blocks[1]
        if hasattr(ub, "attentions"):
            for attn in ub.attentions:
                _disable_self_attention_in_transformer(attn)

    return unet

