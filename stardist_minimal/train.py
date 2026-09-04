from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StarDistDataset
from .losses import StarDistLoss
from .model import StarDist
from .runtime import load_checkpoint, resolve_device, save_checkpoint


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: StarDist,
    loader: DataLoader,
    criterion: StarDistLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "score": 0.0, "rays": 0.0}
    samples = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        score_target = batch["score"].to(device, non_blocking=True)
        ray_target = batch["rays"].to(device, non_blocking=True)
        foreground = batch["foreground"].to(device, non_blocking=True)
        batch_size = image.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with torch.set_grad_enabled(training), context:
            score_pred, ray_pred = model(image)
            losses = criterion(
                score_pred, ray_pred, score_target, ray_target, foreground
            )

        if training:
            assert scaler is not None
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()

        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        samples += batch_size

    return {key: value / max(samples, 1) for key, value in totals.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train single-class 2D StarDist")
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-masks", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-masks", required=True)
    parser.add_argument("--output-dir", default="runs/stardist")
    parser.add_argument("--encoder", default="efficientnet_b5")
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = load_checkpoint(args.resume, device) if args.resume else None
    if resume is not None:
        previous = resume["config"]
        encoder = str(previous["encoder"])
        n_rays = int(previous["n_rays"])
        patch_size = int(previous["patch_size"])
    else:
        encoder, n_rays, patch_size = args.encoder, args.n_rays, args.patch_size

    config = {
        "encoder": encoder,
        "n_rays": n_rays,
        "patch_size": patch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_images": str(args.train_images),
        "train_masks": str(args.train_masks),
        "val_images": str(args.val_images),
        "val_masks": str(args.val_masks),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_set = StarDistDataset(
        args.train_images,
        args.train_masks,
        n_rays=n_rays,
        patch_size=patch_size,
        training=True,
    )
    val_set = StarDistDataset(
        args.val_images,
        args.val_masks,
        n_rays=n_rays,
        patch_size=patch_size,
        training=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = StarDist(
        n_rays=n_rays,
        encoder_name=encoder,
        pretrained=not args.no_pretrained and resume is None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = StarDistLoss()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_val_loss = 1, float("inf")
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scheduler") is not None:
            scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume["epoch"]) + 1
        best_val_loss = float(resume["best_val_loss"])

    history: list[dict[str, float | int]] = []
    history_path = output_dir / "history.json"
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text())

    print(
        f"device={device} encoder={encoder} n_rays={n_rays} "
        f"train={len(train_set)} val={len(val_set)}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]

        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_score": train_metrics["score"],
            "train_rays": train_metrics["rays"],
            "val_score": val_metrics["score"],
            "val_rays": val_metrics["rays"],
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            config,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                config,
            )

        print(
            f"epoch {epoch:03d} train={train_metrics['loss']:.5f} "
            f"val={val_metrics['loss']:.5f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )


if __name__ == "__main__":
    main()
