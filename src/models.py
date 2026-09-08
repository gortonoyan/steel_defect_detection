import segmentation_models_pytorch as smp

def build_model(model_name: str, encoder_name: str = "resnet34", encoder_weights: str = "imagenet", classes: int = 4):
    """
    Builds and returns a segmentation model based on the given parameters.
    
    Args:
        model_name (str): 'unet', 'fpn', or 'deeplabv3plus'.
        encoder_name (str): Backbone architecture (e.g., 'resnet34', 'se_resnext50_32x4d', 'efficientnet-b3').
        encoder_weights (str): Weights for the backbone ('imagenet' or None).
        classes (int): Number of output classes (defects).
        
    Returns:
        torch.nn.Module: The segmentation model.
    """
    model_name = model_name.lower()
    
    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None,
        )
    elif model_name == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None,
        )
    elif model_name == "deeplabv3plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None,
        )
    else:
        raise ValueError(f"Model {model_name} is not supported. Choose from 'unet', 'fpn', or 'deeplabv3plus'.")
        
    return model
