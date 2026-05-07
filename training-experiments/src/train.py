#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.dataset import (
    KAGGLE_SLUG,
    download_kaggle_dataset,
    load_training_dataset,
    resolve_version_root,
)
from src.models import MDPaperLikeConfig, create_md_paper_like_unet
from src.sd15_train_utils import (
    encode_latents_sdvae,
    encode_prompts,
    load_sd15_components,
    training_scheduler,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(gpu_id: int) -> torch.device:
    if gpu_id < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu_id >= n:
        raise SystemExit(
            f"Requested --gpu_id {gpu_id} but only {n} CUDA device(s) are visible. "
            f"Try --gpu_id 0 or set CUDA_VISIBLE_DEVICES."
        )
    dev = torch.device(f"cuda:{gpu_id}")
    print(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(dev)}")
    return dev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent diffusion training (CSV tags or ImageFolder).")
    p.add_argument(
        "--backend",
        choices=("sd15", "md_paper_like", "md_ufo_adv"),
        default="sd15",
    )
    p.add_argument("--sd_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--kaggle_download", action="store_true", help=f"Download {KAGGLE_SLUG} via kagglehub.")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--prompt_template", type=str, default="a photo of a {name}")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output_dir", type=str, default="checkpoints")
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_id", type=int, default=1)
    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional: cap number of optimizer steps.",
    )

    p.add_argument("--teacher_checkpoint", type=str, default=None)

    p.add_argument("--save_every_epochs", type=int, default=1)
    p.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Save every N steps. 0 disables.",
    )

    p.add_argument(
        "--one_step_t",
        type=int,
        default=999,
        help="Fixed timestep used for MD-UFO one-step training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.gpu_id)

    if args.data_root:
        data_path = Path(args.data_root)
    elif args.kaggle_download:
        base = Path(download_kaggle_dataset())
        data_path = resolve_version_root(base)
        print(f"Kaggle hub path: {base}\nUsing version root: {data_path}")
    else:
        raise SystemExit("Provide --data_root or --kaggle_download.")

    ds = load_training_dataset(Path(data_path), args.image_size, args.prompt_template)
    if args.max_samples is not None and args.max_samples < len(ds):
        indices = torch.randperm(len(ds))[: args.max_samples].tolist()
        ds = Subset(ds, indices)

    print(f"Training on {len(ds)} images")

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        prompts = [b[1] for b in batch]
        return imgs, prompts

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )

    if args.image_size % 8 != 0:
        raise SystemExit("--image_size must be divisible by 8.")

    noise_scheduler = training_scheduler(args.num_train_timesteps, args.beta_schedule)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "sd15":
        lr = args.lr if args.lr is not None else 1e-5
        save_name = args.save_name or "unet_sd15_finetuned.pt"

        vae, unet, text_encoder, tokenizer = load_sd15_components(args.sd_model_id, device)
        vae.eval()
        text_encoder.eval()
        for m in (vae, text_encoder):
            for p in m.parameters():
                p.requires_grad = False

        opt = torch.optim.AdamW(unet.parameters(), lr=lr)
        for epoch in range(args.epochs):
            unet.train()
            bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for pixel_01, prompts in bar:
                pixel_01 = pixel_01.to(device, non_blocking=True)
                with torch.no_grad():
                    latents = encode_latents_sdvae(vae, pixel_01)
                    ctx = encode_prompts(tokenizer, text_encoder, prompts, device)
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy = noise_scheduler.add_noise(latents, noise, t)
                pred = unet(noisy, t, ctx).sample
                loss = loss_fn(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                bar.set_postfix(loss=f"{loss.item():.4f}")

        torch.save(unet.state_dict(), out_dir / save_name)
        print(f"Saved: {out_dir / save_name}")
        return

    if args.backend == "md_paper_like":
        lr = args.lr if args.lr is not None else 1e-4
        save_name = args.save_name or "unet_md_paper_like.pt"

        vae, _, text_encoder, tokenizer = load_sd15_components(args.sd_model_id, device)
        vae.eval()
        text_encoder.eval()
        for m in (vae, text_encoder):
            for p in m.parameters():
                p.requires_grad = False

        latent_size = args.image_size // 8
        unet = create_md_paper_like_unet(MDPaperLikeConfig(sample_size=latent_size)).to(device)
        opt = torch.optim.AdamW(unet.parameters(), lr=lr)

        for epoch in range(args.epochs):
            unet.train()
            bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
            for pixel_01, prompts in bar:
                pixel_01 = pixel_01.to(device, non_blocking=True)
                with torch.no_grad():
                    latents = encode_latents_sdvae(vae, pixel_01)
                    ctx = encode_prompts(tokenizer, text_encoder, prompts, device)
                noise = torch.randn_like(latents)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy = noise_scheduler.add_noise(latents, noise, t)
                pred = unet(noisy, t, ctx).sample
                loss = loss_fn(pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                bar.set_postfix(loss=f"{loss.item():.4f}")

            if args.save_every_epochs and ((epoch + 1) % args.save_every_epochs == 0):
                ckpt = out_dir / f"{Path(save_name).stem}_epoch{epoch + 1}.pt"
                torch.save(unet.state_dict(), ckpt)
                print(f"Saved epoch checkpoint: {ckpt}")

        torch.save(unet.state_dict(), out_dir / save_name)
        print(f"Saved: {out_dir / save_name}")
        return

    if args.backend == "md_ufo_adv":
        lr = args.lr if args.lr is not None else 1e-5
        save_name = args.save_name or "unet_md_ufo_adv.pt"
        if not args.teacher_checkpoint:
            raise SystemExit("--teacher_checkpoint required for md_ufo_adv (init checkpoint for G and D)")

        vae, _, text_encoder, tokenizer = load_sd15_components(args.sd_model_id, device)
        vae.eval()
        text_encoder.eval()
        for m in (vae, text_encoder):
            for p in m.parameters():
                p.requires_grad = False

        init_state = torch.load(Path(args.teacher_checkpoint).expanduser().resolve(), map_location=device)

        latent_size = args.image_size // 8
        G = create_md_paper_like_unet(MDPaperLikeConfig(sample_size=latent_size)).to(device)
        D = create_md_paper_like_unet(MDPaperLikeConfig(sample_size=latent_size)).to(device)
        G.load_state_dict(init_state, strict=True)
        D.load_state_dict(init_state, strict=True)

        optG = torch.optim.AdamW(G.parameters(), lr=lr)
        optD = torch.optim.AdamW(D.parameters(), lr=lr)

        t_val = int(args.one_step_t)
        if not (0 <= t_val < args.num_train_timesteps):
            raise SystemExit("--one_step_t must be within [0, num_train_timesteps).")

        it = iter(loader)

        def next_batch():
            nonlocal it
            try:
                return next(it)
            except StopIteration:
                it = iter(loader)
                return next(it)

        def agg_logits(x: torch.Tensor) -> torch.Tensor:
            return x.mean(dim=(1, 2, 3))

        max_steps = args.max_steps if args.max_steps is not None else (args.epochs * len(loader))
        bar = tqdm(range(max_steps), desc=f"md_ufo_adv steps (t={t_val})")

        bce = nn.BCEWithLogitsLoss()

        for step in bar:
            pixel_01, prompts = next_batch()
            pixel_01 = pixel_01.to(device, non_blocking=True)

            with torch.no_grad():
                x0 = encode_latents_sdvae(vae, pixel_01)
                ctx = encode_prompts(tokenizer, text_encoder, prompts, device)
                noise = torch.randn_like(x0)
                t = torch.full((x0.shape[0],), t_val, device=device, dtype=torch.long)
                xt = noise_scheduler.add_noise(x0, noise, t)

            eps = G(xt, t, ctx).sample
            a_t = noise_scheduler.alphas_cumprod.to(device)[t].view(-1, 1, 1, 1)
            x0_pred = (xt - (1.0 - a_t).sqrt() * eps) / a_t.sqrt()

            with torch.no_grad():
                t_prev = torch.clamp(t - 1, min=0)
                x_pred_tprev = noise_scheduler.add_noise(x0_pred.detach(), noise, t_prev)

            with torch.no_grad():
                x_real_tprev = noise_scheduler.add_noise(x0, noise, t_prev)

            D.train()
            optD.zero_grad()
            logit_real = agg_logits(D(x_real_tprev, t_prev, ctx).sample)
            logit_fake = agg_logits(D(x_pred_tprev, t_prev, ctx).sample)
            lossD = bce(logit_real, torch.ones_like(logit_real)) + bce(
                logit_fake, torch.zeros_like(logit_fake)
            )
            lossD.backward()
            optD.step()

            G.train()
            optG.zero_grad()
            logit_fake_g = agg_logits(D(x_pred_tprev, t_prev, ctx).sample.detach())
            loss_adv = bce(logit_fake_g, torch.ones_like(logit_fake_g))
            loss_diff = loss_fn(x0_pred, x0)
            lossG = loss_diff + 0.1 * loss_adv
            lossG.backward()
            optG.step()

            bar.set_postfix(lossG=f"{lossG.item():.4f}", diff=f"{loss_diff.item():.4f}", adv=f"{loss_adv.item():.4f}", lossD=f"{lossD.item():.4f}")

            if args.save_every_steps and (step + 1) % int(args.save_every_steps) == 0:
                ckpt = out_dir / f"{Path(save_name).stem}_step{step + 1}.pt"
                torch.save(G.state_dict(), ckpt)
                print(f"Saved step checkpoint: {ckpt}")

        torch.save(G.state_dict(), out_dir / save_name)
        print(f"Saved: {out_dir / save_name}")
        return


if __name__ == "__main__":
    main()

