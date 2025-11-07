#!/usr/bin/env python3
import argparse
import json
import os
import torch
import logging
import copy
import torch.multiprocessing as mp
from dataset.datamodule import TtsDataModule
from shutil import copyfile
from lhotse.utils import fix_random_seed
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from utils.checkpoint import (
    resume_checkpoint,
)
from utils.common import (
    AttributeDict,
    str2bool,
    get_env_info,
    setup_dist,
    setup_logger,
)
from model.mel_ae import mel_ae

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12356,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=11,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="""Checkpoints of pre-trained models, will load it if not None
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp/mel_ae",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.02, help="The base learning rate."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--model-config",
        type=str,
        default="conf/mel_ae_base.json",
        help="The model configuration file.",
    )

    return parser

def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - env_info:  A dict containing information about the environment.

    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "env_info": get_env_info(),
        }
    )

    return params

def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    with open(params.model_config, "r") as f:
        model_config = json.load(f)
    params.update(model_config["model"])
    params.update(model_config["feature"])

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    os.makedirs(f"{params.exp_dir}", exist_ok=True)
    copyfile(src=params.model_config, dst=f"{params.exp_dir}/model.json")
    setup_logger(f"{params.exp_dir}/log/log-train")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    if torch.cuda.is_available():
        params.device = torch.device("cuda", rank)
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    logging.info(params)
    logging.info("About to create model")

    model = mel_ae(**model_config["model"])
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)
    assert params.start_epoch > 0, params.start_epoch
    if params.start_epoch > 1:
        checkpoints = resume_checkpoint(params=params, model=model, model_avg=model_avg)
    model = model.to(params.device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.base_lr,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    if params.start_epoch > 1 and checkpoints is not None:
        # load state_dict for optimizers
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

        # load state_dict for schedulers
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])
    
    datamodule = TtsDataModule(args)
    train_cuts = datamodule.train_libritts_cuts()
    dev_cuts = datamodule.dev_libritts_cuts()
    
    train_dl = datamodule.train_dataloaders(train_cuts)
    valid_dl = datamodule.dev_dataloaders(dev_cuts)

    logging.info("Training started")
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)
        params.cur_epoch = epoch
        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)
        # train_one_epoch(
        #     params=params,
        #     model=model,
        #     model_avg=model_avg,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     train_dl=train_dl,
        #     valid_dl=valid_dl,
        #     scaler=scaler,
        #     tb_writer=tb_writer,
        #     world_size=world_size,
        #     rank=rank,
        # )
        print(f"pretend train_one_epoch over")
        filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
        # save_checkpoint(
        #     filename=filename,
        #     params=params,
        #     model=model,
        #     model_avg=model_avg,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     sampler=train_dl.sampler,
        #     scaler=scaler,
        #     rank=rank,
        # )
        print(f"pretend save_checkpoint over")
        if rank == 0:
            if params.best_train_epoch == params.cur_epoch:
                best_train_filename = params.exp_dir / "best-train-loss.pt"
                copyfile(src=filename, dst=best_train_filename)

            if params.best_valid_epoch == params.cur_epoch:
                best_valid_filename = params.exp_dir / "best-valid-loss.pt"
                copyfile(src=filename, dst=best_valid_filename)
    
    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()

def main():
    parser = get_parser()
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    if args.world_size > 1:
        mp.spawn(run, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)

if __name__ == "__main__":
    main()