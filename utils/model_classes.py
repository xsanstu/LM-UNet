from models import unet_precip_regression_lightning as unet_regr
import lightning.pytorch as pl

#  -> tuple[type[pl.LightningModule], str]
def get_model_class(model_file):
    # This is for some nice plotting
    if "RainNet" in model_file:
        model_name = "RainNet"
        model = unet_regr.RainNet
    elif "UNetDS_Attention_4kpl" in model_file:
        model_name = "UNetDS Attention with 4kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_1kpl" in model_file:
        model_name = "UNetDS Attention with 1kpl"
        model = unet_regr.UNetDS_Attention
    elif "UNetDS_Attention_4CBAMs" in model_file:
        model_name = "UNetDS Attention 4CBAMs"
        model = unet_regr.UNetDS_Attention_4CBAMs
    elif "UNetDS_Attention" in model_file:
        model_name = "SmaAt-UNet"
        model = unet_regr.UNetDS_Attention
    elif "SAR_UNet" in model_file:
        model_name = "SAR_UNet"
        model = unet_regr.SAR_UNet
    elif "VMambaUnet2_PAM" in model_file:
        model_name = "VMambaUnet2_PAM"
        model = unet_regr.VMambaUnet2_PAM
    elif "LMUNet_upConv" in model_file:
        model_name = "LMUNet_upConv"
        model = unet_regr.LMUNet_upConv
    elif "VMambaUnet3_down" in model_file:
        model_name = "VMambaUnet3_down"
        model = unet_regr.VMambaUnet3_down
    elif "VMambaUnet3" in model_file:
        model_name = "VMambaUnet3"
        model = unet_regr.VMambaUnet3
    elif "LightM" in model_file:
        model_name = "LightMUnet"
        model = unet_regr.LightMUnet
    elif "SSA_UNet" in model_file:
        model_name = "SSA_UNet"
        model = unet_regr.SSA_UNet
    elif "UNet" in model_file:
        model_name = "UNet"
        model = unet_regr.UNet
    else:
        raise NotImplementedError("Model not found")
    return model, model_name
