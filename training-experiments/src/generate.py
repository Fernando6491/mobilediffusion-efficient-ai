#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from PIL import Image

from src.models import MDPaperLikeConfig, create_md_paper_like_unet
from src.sd15_train_utils import encode_prompts, load_sd15_components


def pick_device(gpu_id: int) -> torch.device:
    if gpu_id < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu_id >= n:
        raise SystemExit(f"Requested --gpu_id {gpu_id}, but only {n} CUDA device(s) visible.")
    dev = torch.device(f"cuda:{gpu_id}")
    print(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(dev)}")
    return dev


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    image = (image * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def build_scheduler(
    name: str,
    *,
    num_train_timesteps: int,
    beta_schedule: str,
):
    name = name.lower()
    if name == "ddpm":
        return DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
    if name == "ddim":
        return DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule, clip_sample=False)
    if name in ("dpmpp", "dpm++", "dpm_solver++"):
        return DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
    if name in ("euler_a", "euler-ancestral"):
        return EulerAncestralDiscreteScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=beta_schedule)
    raise SystemExit(f"Unknown --scheduler {name}. Choose from: ddpm, ddim, dpmpp, euler_a")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--backend", choices=("sd15", "md_paper_like", "md_ufo_adv"), default="sd15")
    p.add_argument("--sd_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--prompt", type=str, default="mountain by the lake")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument(
        "--one_step",
        action="store_true",
        help="If set, run a single denoise step at --one_step_t (intended for md_ufo_adv checkpoints).",
    )
    p.add_argument("--one_step_t", type=int, default=999, help="Fixed timestep used for --one_step.")
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        help="Sampling scheduler: ddim (recommended), dpmpp, euler_a, or ddpm.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="checkpoint-test")
    p.add_argument("--output_name", type=str, default="out.png")
    p.add_argument("--print_timing", action="store_true")
    p.add_argument(
        "--print_vram",
        action="store_true",
        help="If set, prints peak GPU VRAM (GiB) during denoise+decode.",
    )
    args = p.parse_args()

    if args.image_size % 8 != 0:
        raise SystemExit("--image_size must be divisible by 8.")

    device = pick_device(args.gpu_id)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0_total = time.perf_counter()
    latent_h = args.image_size // 8
    latent_w = args.image_size // 8

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    scheduler = build_scheduler(
        args.scheduler,
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
    )
    if not args.one_step:
        scheduler.set_timesteps(args.num_inference_steps, device=device)

    t0_load = time.perf_counter()
    vae, sd_unet, text_encoder, tokenizer = load_sd15_components(args.sd_model_id, device)
    vae.eval()
    text_encoder.eval()

    if args.backend == "sd15":
        sd_unet.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
        model = sd_unet
    else:
        model = create_md_paper_like_unet(MDPaperLikeConfig(sample_size=latent_h)).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)

    model.eval()
    ctx = encode_prompts(tokenizer, text_encoder, [args.prompt], device)

    latents = torch.randn((1, 4, latent_h, latent_w), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)
    _sync_cuda_if_needed(device)
    t0_denoise = time.perf_counter()
    if args.one_step:
        t_val = int(args.one_step_t)
        if not (0 <= t_val < args.num_train_timesteps):
            raise SystemExit("--one_step_t must be within [0, num_train_timesteps).")
        t = torch.full((latents.shape[0],), t_val, device=device, dtype=torch.long)
        noise_pred = model(latents, t, ctx).sample
        a_t = scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
        latents = (latents - (1.0 - a_t).sqrt() * noise_pred) / a_t.sqrt()
    else:
        for t in scheduler.timesteps:
            noise_pred = model(latents, t.repeat(latents.shape[0]), ctx).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
    _sync_cuda_if_needed(device)
    t1_denoise = time.perf_counter()

    t0_decode = time.perf_counter()
    dec = vae.decode(latents / 0.18215).sample
    _sync_cuda_if_needed(device)
    t1_decode = time.perf_counter()

    t1_load = time.perf_counter()
    dec = (dec / 2 + 0.5).clamp(0, 1)
    image = tensor_to_pil(dec[0])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    image.save(out_path)
    t1_total = time.perf_counter()
    print(f"Saved generated image: {out_path.resolve()}")

    if args.print_timing:
        print(
            "Timing (wall): "
            f"load={t1_load - t0_load:.2f}s, denoise={t1_denoise - t0_denoise:.2f}s, "
            f"decode={t1_decode - t0_decode:.2f}s, total={t1_total - t0_total:.2f}s"
        )
    if args.print_vram:
        if device.type != "cuda":
            print("VRAM (peak): n/a (CPU)")
        else:
            peak = torch.cuda.max_memory_allocated(device=device) / (1024**3)
            print(f"VRAM (peak, denoise+decode): {peak:.2f} GiB")


if __name__ == "__main__":
    main()

