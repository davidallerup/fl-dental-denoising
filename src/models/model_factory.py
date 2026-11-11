from . import unet, residual_unet

def get_model(model_name: str):

    if model_name == "unet":
        return unet.UNet()
    elif model_name == "residual_unet":
        return residual_unet.Residual_UNet()
    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: 'unet', 'residual_unet'.")