"""
Train a JAX/Flax diffusion model (DiT backbone) with pmap on multi-device (e.g., TPU v5e-8).

This script mirrors the structure of main.py but targets vfsddpm_jax.
"""

import argparse
import time

import jax
import numpy as np

from dataset import create_loader
from model import select_model  # keeps existing namespace for non-JAX
from model.select_model_jax import select_model_jax
from model.vfsddpm_jax import vfsddpm_loss
from model.set_diffusion import logger
from model.set_diffusion.train_util_jax import (
    create_train_state_pmap,
    shard_batch,
    train_step_pmap,
)
from model.set_diffusion.script_util_jax import (
    add_dict_to_argparser as add_dict_to_argparser_jax,
    args_to_dict as args_to_dict_jax,
    model_and_diffusion_defaults as model_and_diffusion_defaults_jax,
)
from utils.path import set_folder
from utils.util import set_seed


DIR = set_folder()


def numpy_from_torch(batch):
    # Assume batch is a torch Tensor on CPU; values in [0,1] -> scale to [-1,1]
    arr = batch.detach().cpu().numpy().astype(np.float32)
    if arr.max() <= 1.01 and arr.min() >= -0.01:
        arr = arr * 2.0 - 1.0
    return arr


def main():
    args = create_argparser().parse_args()
    set_seed(getattr(args, "seed", 0))

    print("\nArgs:")
    for k in sorted(vars(args)):
        print(k, getattr(args, k))
    print()

    logger.configure(dir=DIR, mode="training_jax", args=args, tag="jax")

    n_devices = jax.local_device_count()
    if args.batch_size % n_devices != 0:
        raise ValueError(f"batch_size {args.batch_size} must be divisible by n_devices {n_devices}")

    rng = jax.random.PRNGKey(getattr(args, "seed", 0))
    rng, rng_model = jax.random.split(rng)

    params, modules, cfg = select_model_jax(args, rng_model)

    # Train state and optimizer
    state, tx = create_train_state_pmap(
        params,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    p_state = jax.device_put_replicated(state, jax.local_devices())

    def loss_fn(p, batch, rng_in):
        return vfsddpm_loss(rng_in, p, modules, batch, cfg, train=True)

    p_train_step = train_step_pmap(tx, loss_fn, ema_rate=float(str(args.ema_rate).split(",")[0]))

    # Data loaders (PyTorch) on CPU
    train_loader = create_loader(args, split="train", shuffle=True)

    logger.log("starting training (jax pmap)...")
    global_step = 0
    for epoch in range(10**6):  # effectively infinite unless steps reached
        for batch in train_loader:
            batch_np = numpy_from_torch(batch)
            try:
                batch_sharded = shard_batch(batch_np, n_devices)
            except AssertionError:
                # skip incomplete batch
                continue

            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, n_devices)

            p_state, metrics = p_train_step(p_state, batch_sharded, step_rngs)

            # host metrics
            metrics_host = jax.tree_map(lambda x: np.array(x).mean(), metrics)
            global_step += 1

            if global_step % args.log_interval == 0:
                logger.logkv("step", global_step)
                for k, v in metrics_host.items():
                    if isinstance(v, np.ndarray):
                        v = v.item() if v.size == 1 else v
                    logger.logkv_mean(k, v)
                logger.dumpkvs()

            if args.lr_anneal_steps and global_step >= args.lr_anneal_steps:
                break
        if args.lr_anneal_steps and global_step >= args.lr_anneal_steps:
            break

    logger.log("training complete.")


def create_argparser():
    defaults = dict(
        model="vfsddpm_jax",
        dataset="cifar100",
        image_size=32,
        sample_size=5,
        patch_size=8,
        hdim=256,
        in_channels=3,
        encoder_mode="vit_set",
        pool="cls",
        context_channels=256,
        mode_context="deterministic",
        mode_conditioning="film",
        augment=False,
        data_dir="/home/gigi/ns_data",
        num_classes=1,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        log_interval=100,
        ema_rate="0.9999",
        resume_checkpoint="",
        clip_denoised=True,
        use_ddim=False,
        tag=None,
        seed=0,
    )
    defaults.update(model_and_diffusion_defaults_jax())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser_jax(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

