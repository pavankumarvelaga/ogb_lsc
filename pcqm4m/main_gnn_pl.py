import argparse
import gc
import os
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path
<<<<<<< HEAD

=======
import shutil
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from ogb.lsc import PCQM4MEvaluator, PygPCQM4MDataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from gnn import GNN

<<<<<<< HEAD
=======
from pytorch_lightning.plugins import DDPSpawnPlugin


>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557

class PCQM4MRegressor(pl.LightningModule):
    def __init__(
        self,
        gnn: str = "gin-virtual",
        num_layers: int = 5,
        emb_dim: int = 600,
        drop_ratio: float = 0.0,
        graph_pooling: str = "max",
        learning_rate: float = 0.001,
        step_size: int = 30,
        gamma: float = 0.25,
    ):
        super().__init__()
        self.save_hyperparameters()

        shared_params = {
            "num_layers": num_layers,
            "emb_dim": emb_dim,
            "drop_ratio": drop_ratio,
            "graph_pooling": graph_pooling,
        }

        if gnn == "gin":
            self.model = GNN(gnn_type="gin", virtual_node=False, **shared_params)
        elif gnn == "gin-virtual":
            self.model = GNN(gnn_type="gin", virtual_node=True, **shared_params)
        elif gnn == "gcn":
            self.model = GNN(gnn_type="gcn", virtual_node=False, **shared_params)
        elif gnn == "gcn-virtual":
            self.model = GNN(gnn_type="gcn", virtual_node=True, **shared_params)
        else:
            raise ValueError("Invalid GNN type")

        self.reg_criterion = torch.nn.L1Loss()

    def forward(self, x):
        # use forward for inference/predictions
        predictions = self.model(x).view(
            -1,
        )
        return predictions

    def training_step(self, batch, batch_idx):
        pred = self.model(batch).view(
            -1,
        )
        loss = self.reg_criterion(pred, batch.y)
<<<<<<< HEAD
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
=======
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch).view(
            -1,
        )
        loss = self.reg_criterion(pred, batch.y)
<<<<<<< HEAD
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
=======
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return {
            "optimizer": optimizer,
<<<<<<< HEAD
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
=======
            "lr_scheduler":  scheduler
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
        }


class PCQM4MDataModule(pl.LightningDataModule):
    def __init__(
        self, train_subset: bool = False, batch_size: int = 256, num_workers: int = 8
    ):
        super().__init__()

        dataset = PygPCQM4MDataset(root="../../data")
        split_idx = dataset.get_idx_split()

        if train_subset:
            subset_ratio = 0.1
            subset_idx = torch.randperm(len(split_idx["train"]))[
                : int(subset_ratio * len(split_idx["train"]))
            ]
            self.train_dataset = dataset[split_idx["train"][subset_idx]]
        else:
            self.train_dataset = dataset[split_idx["train"]]

        self.valid_dataset = dataset[split_idx["valid"]]

        self.test_dataset = dataset[split_idx["test"]]

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
<<<<<<< HEAD
=======
            persistent_workers=True
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
<<<<<<< HEAD
=======
            persistent_workers=True
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
<<<<<<< HEAD
=======
            persistent_workers=True
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
        )


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(SEED):
    parser = argparse.ArgumentParser(
        description=(
            "GNN baselines on pcqm4m with Pytorch Geometrics and Pytorch Lightning"
        )
    )
    parser.add_argument(
        "--gnn",
        type=str,
        default="gin-virtual",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=600,
        help="dimensionality of hidden units in GNNs (default: 600)",
    )
    parser.add_argument(
        "--drop_ratio", type=float, default=0, help="dropout ratio (default: 0)"
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="sum",
        help="graph pooling strategy mean or sum (default: sum)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate for optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=30,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument("--train_subset", action="store_false")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of workers (default: 0)"
    )
    parser.add_argument("--gpu_ids", type=str, default="1", help="GPU ids (default: 1)")
    args = parser.parse_args()

    if args.train_subset:
        args.step_size *= 10

    print(args)

    seed_everything(SEED)

<<<<<<< HEAD
    log_path = "/home/pv217/kddcup21/ogb_lsc/pl_pcqm4m_logs"
    checkpoint_path = "/home/pv217/kddcup21/ogb_lsc/pl_pcqm4m_checkpoints"
=======
    log_path = Path("/home/pv217/kddcup21/ogb_lsc/pl_pcqm4m_logs")
    checkpoint_path = Path("/home/pv217/kddcup21/ogb_lsc/pl_pcqm4m_checkpoints")

    if log_path.exists():
        shutil.rmtree(str(log_path))
    if checkpoint_path.exists():
        shutil.rmtree(str(checkpoint_path))
    log_path.mkdir(exist_ok=True)
    checkpoint_path.mkdir(exist_ok=True)
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557

    dm = PCQM4MDataModule(args.train_subset, args.batch_size, args.num_workers)
    dm.setup()
    model = PCQM4MRegressor(
        gnn=args.gnn,
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.graph_pooling,
        learning_rate=args.learning_rate,
        step_size=args.step_size,
        gamma=args.gamma,
    )
    testtube_logger = TestTubeLogger(str(log_path), name="PCQM4M", version=0)
<<<<<<< HEAD
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{checkpoint_path}/pcqm4m_" + "{epoch:02d}-{val_loss_epoch:.4f}",
=======
    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch:02d}-{val_loss_epoch:.4f}",
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
        save_top_k=1,
        verbose=True,
        monitor="val_loss_epoch",
        mode="min",
    )
    trainer = pl.Trainer(
        weights_summary="top",
        num_sanity_val_steps=0,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
<<<<<<< HEAD
        checkpoint_callback=checkpoint_callback,
        logger=testtube_logger,
        gpus=args.gpu_ids,
=======
        checkpoint_callback=True,
        callbacks=[ckpt_callback],
        logger=testtube_logger,
        gpus=args.gpu_ids,
        accelerator="ddp_spawn",
        plugins=DDPSpawnPlugin(find_unused_parameters=False),
>>>>>>> 858a67fc9a656ab5bb4b98513603d69118f4f557
        deterministic=True,
        precision=32,
        # fast_dev_run=True,
    )
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    # val_probs = predictor(model.model, dm.val_dataloader())
    # test_probs = predictor(model.model, dm.test_dataloader())


if __name__ == "__main__":
    main(1729)
