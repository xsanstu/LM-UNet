from models.unet_precip_regression_lightning import SSA_UNet
from root import ROOT_DIR

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch import loggers
import argparse
from models import unet_precip_regression_lightning as unet_regr
from lightning.pytorch.tuner import Tuner


def train_regression(hparams, find_batch_size_automatically: bool = False):
    if hparams.model == "UNet":
        net = unet_regr.UNet(hparams=hparams)
    elif hparams.model == "LM_UNet":
        net = unet_regr.LM_UNet(hparams=hparams)
    else:
        raise NotImplementedError(f"Model '{hparams.model}' not implemented")

    default_save_path = ROOT_DIR / "lightning" / "precip_regression"

    checkpoint_callback = ModelCheckpoint(
        dirpath=default_save_path / net.__class__.__name__,
        filename=net.__class__.__name__ + "_rain_threshold_20_{epoch}-{val_loss:.6f}",
        save_top_k=-1,
        verbose=False,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor()
    tb_logger = loggers.TensorBoardLogger(save_dir=default_save_path, name=net.__class__.__name__)

    earlystopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=hparams.es_patience,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        default_root_dir=default_save_path,
        logger=tb_logger,
        callbacks=[checkpoint_callback, earlystopping_callback, lr_monitor],
        val_check_interval=hparams.val_check_interval,
    )

    if find_batch_size_automatically:
        tuner = Tuner(trainer)

        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(net, mode="binsearch")

    trainer.fit(model=net, ckpt_path=hparams.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建一个参数解析器

    # 重要
    parser = unet_regr.Precip_regression_base.add_model_specific_args(parser)

    parser.add_argument(
        "--dataset_folder",
        default=ROOT_DIR / "data" / "precipitation" / "RAD_NL25_RAC_5min_train_test_2016-2019.h5",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--fast_dev_run", type=bool, default=False)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--val_check_interval", type=float, default=None)
    # 解析添加的参数
    args = parser.parse_args()

    args.n_channels = 12   ###################################################################   注意   ######
    args.n_classes = 1
    # args.model = "UNet"
    args.model = "LM_UNet"
    args.lr_patience = 4

    args.es_patience = 15
    # args.val_check_interval = 0.25
    args.kernels_per_layer = 2
    args.use_oversampled_dataset = True
    args.dataset_folder = (
        # /KNMI/train_test_2016-2019_input-length_24_img-ahead_6_rain-threshold_50.h5"
        "/KNMI/train_test_2016-2019_input-length_24_img-ahead_6_rain-threshold_20.h5"
    )
    train_regression(args, find_batch_size_automatically=False)

