import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from core.model import jade
from core import optimize
from utils import checkpoint
from utils.logging import get


def select_device_and_precision():
    if not torch.cuda.is_available():
        return "cpu", torch.float32
    major, _ = torch.cuda.get_device_capability()
    precision = torch.bfloat16 if major >= 8 else torch.float16
    return "cuda", precision


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return 1, 0, 0
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return world_size, rank, local_rank


def load_tokens(path):
    path = Path(path)
    if path.is_dir():
        shard_paths = sorted(path.glob("*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"no .bin token shards found in {path}")
    else:
        shard_paths = [path]
    return [np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths]


def get_batch(shards, batch_size, context, device):
    lengths = np.clip(np.array([len(s) - context - 1 for s in shards]), 0, None)
    weights = lengths / lengths.sum()
    shard_indices = np.random.choice(len(shards), size=batch_size, p=weights)

    x_list, y_list = [], []
    for shard_index in shard_indices:
        tokens = shards[shard_index]
        start = np.random.randint(0, len(tokens) - context - 1)
        x_list.append(torch.from_numpy(tokens[start:start + context].astype(np.int64)))
        y_list.append(torch.from_numpy(tokens[start + 1:start + 1 + context].astype(np.int64)))

    x = torch.stack(x_list)
    y = torch.stack(y_list)
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def lr_scale(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(model_config, train_config, token_dir, checkpoint_dir, experiment, git_commit, repo_id=None, log_path=None):
    logger = get()
    world_size, rank, local_rank = setup_distributed()
    device_type, precision = select_device_and_precision()
    device = f"cuda:{local_rank}" if device_type == "cuda" else "cpu"

    torch.manual_seed(42 + rank)

    model = jade(model_config).to(device)
    raw_model = model

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[local_rank] if device_type == "cuda" else None)

    try:
        model = torch.compile(model)
    except Exception:
        pass

    optimizers = optimize.build_optimizer(raw_model, train_config)
    base_lrs = [group["lr"] for optimizer in optimizers for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == torch.float16))

    tokens = load_tokens(token_dir)
    step = 0

    restored = checkpoint.restore(checkpoint_dir, repo_id=repo_id)
    if restored:
        raw_model.load_state_dict(restored["model"])
        step = restored["step"]
        logger.info(f"resumed from step {step}")

    start_time = time.time()
    tokens_per_step = train_config["batch_size"] * model_config["context"] * train_config["grad_accum"] * world_size
    total_tokens = step * tokens_per_step

    while step < train_config["steps"]:
        scale = lr_scale(step, train_config["warmup"], train_config["steps"])
        index = 0
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = base_lrs[index] * scale
                index += 1

        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0
        for _ in range(train_config["grad_accum"]):
            x, y = get_batch(tokens, train_config["batch_size"], model_config["context"], device)
            with torch.autocast(device_type=device_type, dtype=precision, enabled=(device_type == "cuda")):
                _, loss = model(x, y)
                loss = loss / train_config["grad_accum"]
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()

        for optimizer in optimizers:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_config["grad_clip"])
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.update()

        step += 1
        total_tokens += tokens_per_step

        if rank == 0 and step % 10 == 0:
            elapsed = time.time() - start_time
            throughput = total_tokens / max(elapsed, 1e-9)
            logger.info(f"step {step}/{train_config['steps']}  loss {accumulated_loss:.4f}  lr {scale * base_lrs[0]:.6f}  tokens/s {throughput:.0f}")

        if rank == 0 and step % train_config["checkpoint_every"] == 0:
            state = checkpoint.build(
                model_state=raw_model.state_dict(),
                optimizer_state=[o.state_dict() for o in optimizers],
                scheduler_state={"step": step},
                rng_state=torch.get_rng_state(),
                step=step,
                epoch=1,
                tokens=total_tokens,
                config=model_config,
                tokenizer_info={"name": "gpt2-bootstrap"},
                dataset_info={"token_dir": str(token_dir)},
                experiment=experiment,
                commit=git_commit,
            )
            ok, path = checkpoint.commit(state, step, checkpoint_dir, repo_id=repo_id, log_path=log_path)
            logger.info(f"checkpoint step {step} verified={ok} path={path}")

    return raw_model

