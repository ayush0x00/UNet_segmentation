import torch
import torch.nn as nn
import torchvision.transforms.functional as T
from torchvision import transforms
from PIL import Image



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
    ) -> None:  # out channels is 1 beacause we will do binary segmentation
        super(UNET,self).__init__()

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.down.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.up.append(nn.ConvTranspose2d(
            in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2
        ))  # feature * 2 as we will be appending the skip connection along the channel dim)
            self.up.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []
        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.up),2):
            x = self.up[idx](x)
            skip = skip_connections[idx//2]

            if(x.shape != skip.shape):
                skip = T.center_crop(skip,x.shape[2:])

            concat_skip = torch.cat((skip,x),dim=1)
            x = self.up[idx+1](concat_skip)

        return self.final_conv(x)

# def test():
#     # x= torch.randn((3,3,160,160))
#     t_img = Image.open("./test.jpg").convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize((160,160)),
#         transforms.ToTensor()
#     ])
    
#     tran_img = transform(t_img)
#     print(tran_img.shape)
#     model = UNET(in_channels=3, out_channels=3)
#     preds = model(tran_img.unsqueeze(0))

#     img_pil = T.to_pil_image(preds[0])
#     img_pil.show()

    

# if __name__=='__main__':
#     test()
