import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from mamba_ssm import Mamba

def reshape_patch(img_tensor, patch_size):
    # assert 5 == img_tensor.ndim
    batch_size = img_tensor.shape[0]
    seq_length = img_tensor.shape[1]
    img_height = img_tensor.shape[2]
    img_width = img_tensor.shape[3]
    # num_channels = img_tensor.shape[0][4]
    a = img_tensor.reshape([batch_size, seq_length,
                            img_height//patch_size, patch_size,
                            img_width//patch_size, patch_size,
                            # num_channels
                            ])
    b = a.permute([0,1,2,4,3,5])
    patch_tensor = b.reshape([batch_size, seq_length,
                            img_height//patch_size,
                            img_width//patch_size,
                            patch_size*patch_size  #*num_channels
                              ])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    # assert 5 == patch_tensor.ndim
    batch_size = patch_tensor.shape[0]
    seq_length = patch_tensor.shape[1]
    patch_height = patch_tensor.shape[2]
    patch_width = patch_tensor.shape[3]
    # channels = np.shape(patch_tensor)[4]
    # img_channels = channels // (patch_size*patch_size)
    a = patch_tensor.reshape([batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  # img_channels
                              ])
    # b = np.transpose(a, [0,1,2,4,3,5,6])
    b = a.permute([0,1,2,4,3,5])
    img_tensor = b.reshape([batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                # img_channels
                            ])
    return img_tensor

class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)        # why not cov
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):               # x:[16, 32, 288,288]这个实现为RVMlayer
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        x = reshape_patch(x,patch_size=4)
        B, C = x.shape[:2]              # 表示从数组的开头到索引为2的位置之前的所有元素B:batch_size C:特征图通道
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()  # 2维情况下[288*288=82944]表示从数组的索引为2到末尾的位置之前的所有元素W H D  这里是三个相乘W*H*D的总数量
        img_dims = x.shape[2:]          # 2维情况下[288，288]图像的尺寸W H D
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)        # 2维情况下[16, 82944, 32]扁平化（B,n_tokens,C），扁平化不就损失了图像空间特征么
        x_norm = self.norm(x_flat)                                  # 归一化
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat     # mamba后(VSS)的结果+论文里的尺度因子
        x_mamba = self.norm(x_mamba)                                # 归一化
        x_mamba = self.proj(x_mamba)                                # 线性
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)  # 将数据恢复为原始的B C W H D
        out = reshape_patch_back(out, patch_size=4)
        return out


class ResMambaBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        # act: tuple | str = ("RELU", {"inplace": True}),
        act: tuple = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            # norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """
        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_mamba_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)
        self.conv2 = get_mamba_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels)

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity
        return x

class ResUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        # act: tuple | str = ("RELU", {"inplace": True}),
        act: tuple = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity       # 这不是残差么
        x = self.norm2(x)
        x = self.act(x)
        return x

def get_dwconv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)


def get_mamba_layer(spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1):
    mamba_layer = MambaLayer(input_dim=in_channels, output_dim=out_channels)        # RVMlayer 根据论文中间应该是有VSS，程序中写的是mamba应该是一个意思吧
    if stride != 1:
        if spatial_dims==2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims==3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


def _make_up_layers(self):
    up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
    upsample_mode, blocks_up, spatial_dims, filters, norm = (
        self.upsample_mode,
        self.blocks_up,
        self.spatial_dims,
        self.init_filters,
        self.norm,
    )
    n_up = len(blocks_up)
    for i in range(n_up):
        sample_in_channels = filters * 2 ** (n_up - i)
        up_layers.append(
            nn.Sequential(
                *[
                    ResUpBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                    for _ in range(blocks_up[i])
                ]
            )
        )
        up_samples.append(
            nn.Sequential(
                *[
                    get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                    get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                ]
            )
        )
    return up_layers, up_samples
