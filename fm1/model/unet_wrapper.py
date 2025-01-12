from .unet_openai import UNetModel, torch, nn




class ModelUnet(nn.Module):
    def __init__(self, modelConf):
        super().__init__()
        self.unet = UNetModel(
            image_size=64,
            in_channels=3,
            model_channels=32,
            out_channels=3,
            num_res_blocks=2,
            # attention_resolutions=[],
            attention_resolutions=[8],
        )

    def forward(self, x,t):
        return self.unet(x,t.squeeze())
