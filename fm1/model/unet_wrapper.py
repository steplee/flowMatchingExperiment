import torch, torch.nn as nn




class ModelUnet(nn.Module):
    def __init__(self, modelConf):
        super().__init__()

        clazz = None
        version = modelConf.get('version', 1)

        if version == 1:
            from .unet_openai import UNetModel
        elif version == 2:
            from .unet_openai2 import UNetModel

        self.unet = UNetModel(
            image_size=modelConf.imgSize,
            in_channels=3,
            model_channels=32,
            out_channels=3,
            num_res_blocks=2,
            # attention_resolutions=[],
            attention_resolutions=[8],
        )

    def forward(self, x,t):
        return self.unet(x,t.squeeze())
