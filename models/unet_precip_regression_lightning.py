import torch
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from monai.networks.layers import get_norm_layer, get_act_layer
from timm.layers import trunc_normal_

from models.block import DANetHead, PAM_Module
from models.unet_parts import Down, DoubleConv, Up, OutConv, ModifiedDepthwiseSeparableConv, \
    DownModifiedDepthwiseSeparableConv
from models.unet_parts_depthwise_separable import DoubleShuffledConvDS,DownShuffledDS, DoubleConvDS, UpDS, DownDS
from models.layers import CBAM, SinusoidalPosEmb
from models.regression_lightning import Precip_regression_base
from einops import repeat

from models.vmamba_parts import VSSMEncoder, UNetResDecoder, PatchExpand, UNetResDecoder2


class UNet(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):   # x.shape(16, 12, 288, 288)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)     # 64
        logits = self.outc(x)
        return logits

class LM_UNet(Precip_regression_base):
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.inc = ModifiedDepthwiseSeparableConv(self.n_channels, 32)
        self.down1 = DownModifiedDepthwiseSeparableConv(32, 64)
        self.down2 = DownModifiedDepthwiseSeparableConv(64, 128)
        self.down3 = DownModifiedDepthwiseSeparableConv(128, 256)
        self.down4 = DownModifiedDepthwiseSeparableConv(256, 512)

        self.decoder = UNetResDecoder2(num_classes=self.n_classes,
                                      features_per_stage=[32, 64, 128, 256, 512])

        self.outc = OutConv(32, self.n_classes)

    def forward(self, x):
        skips = []
        x = self.inc(x)       # 8 32 288 288
        skips.append(x)

        x = self.down1(x)     # 8 64 144 144
        skips.append(x)

        x = self.down2(x)     # 8 128 72 72
        skips.append(x)

        x = self.down3(x)     # 8 256 36 36
        skips.append(x)

        x = self.down4(x)     # 8 512 18 18
        skips.append(x)

        # out = self.decoder(x1, x2, x3, x4)  # 8 64 288 288
        out = self.decoder(skips)  # 8 64 288 288

        logits = self.outc(out)  # 48->1
        return logits

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True












