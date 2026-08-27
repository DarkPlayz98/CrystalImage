import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .config import Config
from .model import CrystalLLM


TOKENS = Path(
    "data/tokens/train.bin"
)

CHECKPOINT_DIR = Path(
    "checkpoints"
)


def setup_ddp():
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    if world_size > 1:
        torch.cuda.set_device(
            local_rank
        )

        dist.init_process_group(
            backend="nccl"
        )

    return (
        world_size,
        rank,
        local_rank,
    )


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_batch(
    data,
    batch_size,
    context_length,
    device,
):
    starts = torch.randint(
        0,
        len(data) - context_length - 1,
        (
            batch_size,
        ),
    )

    x = torch.stack(
        [
            torch.from_numpy(
                data[
                    int(start):
                    int(start)
                    + context_length
                ].astype(
                    np.int64
                )
            )
            for start in starts
        ]
    )

    y = torch.stack(
        [
            torch.from_numpy(
                data[
                    int(start) + 1:
                    int(start) + 1
                    + context_length
                ].astype(
                    np.int64
                )
            )
            for start in starts
        ]
    )

    return (
        x.to(
            device,
            non_blocking=True,
        ),
        y.to(
            device,
            non_blocking=True,
        ),
    )


def learning_rate(
    step,
    config,
):
    if step < config.warmup_steps:
        return (
            config.learning_rate
            * (step + 1)
            / config.warmup_steps
        )

    progress = (
        step - config.warmup_steps
    ) / max(
        1,
        config.max_steps
        - config.warmup_steps,
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine = (
        0.5
        * (
            1.0
            + np.cos(
                np.pi * progress
            )
        )
    )

    return (
        config.min_learning_rate
        + (
            config.learning_rate
            - config.min_learning_rate
        )
        * cosine
    )


def save_checkpoint(
    model,
    optimizer,
    scaler,
    step,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = (
        model.module
        if isinstance(
            model,
            DistributedDataParallel,
        )
        else model
    )

    path = (
        CHECKPOINT_DIR
        / f"crystalllm_step_{step}.pt"
    )

    torch.save(
        {
            "step": step,
            "model": state.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )

    return path


def main():
    (
        world_size,
        rank,
        local_rank,
    ) = setup_ddp()

    try:
        config = Config()

        device = torch.device(
            "cuda",
            local_rank,
        )

        random.seed(
            config.seed + rank
        )

        np.random.seed(
            config.seed + rank
        )

        torch.manual_seed(
            config.seed + rank
        )

        data = np.memmap(
            TOKENS,
            dtype=np.uint16,
            mode="r",
        )

        if rank == 0:
            print("=" * 50)
            print("          CrystalLLM v1.0")
            print("=" * 50)
            print(
                f"Tokens: {len(data):,}"
            )
            print(
                f"Parameters config: "
                f"{config.n_layers} layers / "
                f"{config.d_model} dim"
            )
            print(
                f"GPUs: {world_size}"
            )

        model = CrystalLLM(
            config
        ).to(device)

        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[
                    local_rank
                ],
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        scaler = torch.amp.GradScaler(
            "cuda",
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        progress = range(
            1,
            config.max_steps + 1,
        )

        if rank == 0:
            progress = tqdm(
                progress,
                desc="Training",
            )

        for step in progress:
            lr = learning_rate(
                step - 1,
                config,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            total_loss = 0.0

            for _ in range(
                config.gradient_accumulation
            ):
                x, y = get_batch(
                    data,
                    config.batch_size_per_gpu,
                    config.context_length,
                    device,
                )

                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                ):
                    _, loss = model(
                        x,
                        y,
                    )

                    loss = (
                        loss
                        / config.gradient_accumulation
                    )

                scaler.scale(
                    loss
                ).backward()

                total_loss += (
                    loss.item()
                )

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            if (
                rank == 0
                and step % config.log_every == 0
            ):
                progress.set_postfix(
                    loss=f"{total_loss:.4f}",
                    lr=f"{lr:.2e}",
                )

            if (
                step % config.save_every == 0
            ):
                if rank == 0:
                    saved = save_checkpoint(
                        model,
                        optimizer,
                        scaler,
                        step,
                    )

                    print(
                        f"\nCheckpoint: {saved}"
                    )

                if world_size > 1:
                    dist.barrier()

        if rank == 0:
            save_checkpoint(
                model,
                optimizer,
                scaler,
                config.max_steps,
            )

            print(
                "\nCrystalLLM v1.0 training complete."
            )

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
