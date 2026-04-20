import lightning.pytorch as pl
from torch import nn, optim
# from torch_ssim import ssim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from loss.FBMSE import FBMSELoss
from models.RaninHCNet.tool import CSM_Loss
from utils import dataset_precip
import argparse
import numpy as np


class UNet_base(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model",
            type=str,
            default="UNet",
            choices=["UNet", "UNetDS", "UNet_Attention", "UNetDS_Attention"],
        )
        parser.add_argument("--n_channels", type=int, default=12)
        parser.add_argument("--n_classes", type=int, default=1)          ##            注意这里   ######
        parser.add_argument("--kernels_per_layer", type=int, default=1)
        parser.add_argument("--bilinear", type=bool, default=True)      # bilinear：双线性
        parser.add_argument("--reduction_ratio", type=int, default=16)
        parser.add_argument("--lr_patience", type=int, default=5)
        return parser

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lossFn = FBMSELoss()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.1, patience=self.hparams.lr_patience
            ),
            "monitor": "val_loss",  # Default: val_loss
        }
        return [opt], [scheduler]

    # def loss_func(self, y_pred, y_true):
    #     w, loss = self.lossFn(y_pred, y_true)
    #     return w, loss
    def loss_func(self, y_pred, y_true):
        # reduction="mean" is average of every pixel, but I want average of image
        # return ssim
        return nn.functional.mse_loss(y_pred, y_true, reduction="sum") / y_true.size(0)
        # return CSM_Loss(y_pred, y_true) / y_true.size(0)

        # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_pred = self(x)
    #     w, loss = self.loss_func(y_pred.squeeze(), y)
    #     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     for wi in range(len(w)):
    #         self.log("w"+str(wi), w[wi], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     return loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        if x.size(0) < 8:
            return
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        # loss = self.loss_func(y_pred[0], y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_pred = self(x)
    #     w, loss = self.loss_func(y_pred.squeeze(), y)
    #     self.log("val_loss", loss, prog_bar=True)
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        # loss = self.loss_func(y_pred[0], y)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Calculate the loss (MSE per default) on the test set normalized and denormalized."""
        x, y = batch
        y_pred = self(x)
        loss = self.loss_func(y_pred.squeeze(), y)
        factor = 47.83
        loss_denorm = self.loss_func(y_pred.squeeze() * factor, y * factor)
        self.log("MSE", loss)
        self.log("MSE_denormalized", loss_denorm)


class Precip_regression_base(UNet_base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = UNet_base.add_model_specific_args(parent_parser)
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_input_images", type=int, default=12)
        # parser.add_argument("--num_output_images", type=int, default=6)
        parser.add_argument("--num_output_images", type=int, default=1)  # 这是后来序列又加长了12帧
        parser.add_argument("--valid_size", type=float, default=0.1)
        parser.add_argument("--use_oversampled_dataset", type=bool, default=True)
        parser.n_channels = parser.parse_args().num_input_images  # 12
        parser.n_classes = 1
        return parser

    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_sampler = None
        self.valid_sampler = None

    # 这玩意是pytorch lightning的特性oh
    def prepare_data(self):
        # train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip()]
        # )
        train_transform = None
        valid_transform = None
        precip_dataset = (
            dataset_precip.precipitation_maps_oversampled_h5
            if self.hparams.use_oversampled_dataset
            else dataset_precip.precipitation_maps_h5
        )
        self.train_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,                # 数据集文件夹
            num_input_images=self.hparams.num_input_images,     # 12
            num_output_images=self.hparams.num_output_images,   # 6
            train=True,
            transform=train_transform,
        )
        self.valid_dataset = precip_dataset(
            in_file=self.hparams.dataset_folder,
            num_input_images=self.hparams.num_input_images,     # 12
            num_output_images=self.hparams.num_output_images,   # 6
            train=True,
            transform=valid_transform,
        )

        num_train = len(self.train_dataset)     # train数据集的数量M
        indices = list(range(num_train))
        split = int(np.floor(self.hparams.valid_size * num_train))  # 训练集中划分训练和验证的索引值分界点，就是数量

        np.random.shuffle(indices)      # 在原数组上改变自身序列

        train_idx, valid_idx = indices[split:], indices[:split] # 训练数据的索引值和验证数据的索引值
        self.train_sampler = SubsetRandomSampler(train_idx)     # 随机采样
        self.valid_sampler = SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            pin_memory=True,
            # The following can/should be tweaked depending on the number of CPU cores
            num_workers=1,
            persistent_workers=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self.valid_sampler,
            pin_memory=True,
            # The following can/should be tweaked depending on the number of CPU cores
            num_workers=1,
            persistent_workers=True,
        )
        return valid_loader
