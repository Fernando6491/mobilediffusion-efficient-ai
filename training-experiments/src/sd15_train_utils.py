"""Helpers for Stable Diffusion 1.5–style latent training (frozen VAE + text encoder)."""
from __future__ import annotations

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


def load_sd15_components(
    model_id: str,
    device: torch.device,
) -> tuple[AutoencoderKL, UNet2DConditionModel, CLIPTextModel, CLIPTokenizer]:
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    return vae, unet, text_encoder, tokenizer


@torch.no_grad()
def encode_latents_sdvae(vae: AutoencoderKL, pixel_01: torch.Tensor) -> torch.Tensor:
    """RGB in [0,1] -> SD latents scaled by 0.18215."""
    x = pixel_01 * 2.0 - 1.0
    z = vae.encode(x).latent_dist.sample()
    return z * 0.18215


@torch.no_grad()
def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompts: list[str],
    device: torch.device,
) -> torch.Tensor:
    tok = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_encoder(
        input_ids=tok.input_ids.to(device),
        attention_mask=tok.attention_mask.to(device),
    ).last_hidden_state


def training_scheduler(
    num_train_timesteps: int = 1000,
    beta_schedule: str = "linear",
) -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)

