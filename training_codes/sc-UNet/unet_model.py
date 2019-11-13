# full assembly of the sub-parts to form the complete net
import torch.nn as nn
from unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down11 = down(64, 128)
        self.down21 = down(128, 256)
        self.down31 = down(256, 512)
        self.down41 = down(512, 1024)
        self.up11 = up(1024, 512)
        self.up21 = up(512, 256)
        self.up31 = up(256, 128)
        self.up41 = up(128, 64)
        self.unet_1st_out = outconv(64, n_channels)
        
        self.inc0 = inconv(n_channels*2, 64)
        self.down12 = down(64, 128)
        self.down22 = down(128, 256)
        self.down32 = down(256, 512)
        self.down42 = down(512, 1024)
        self.up12 = up(1024, 512)
        self.up22 = up(512, 256)
        self.up32 = up(256, 128)
        self.up42 = up(128, 64)
        self.unet_2nd_out = outconv(64, n_classes)
        

    def forward(self, x):
        x_in = x
        x1 = self.inc(x)
        x2 = self.down11(x1)
        x3 = self.down21(x2)
        x4 = self.down31(x3)
        x5 = self.down41(x4)
        x = self.up11(x5, x4)
        x = self.up21(x, x3)
        x = self.up31(x, x2)
        x = self.up41(x, x1)
        x = self.unet_1st_out(x)
        x = torch.cat([x_in, x], dim=1)
        x1 = self.inc0(x)
        x2 = self.down12(x1)
        x3 = self.down22(x2)
        x4 = self.down32(x3)
        x5 = self.down42(x4)
        x = self.up12(x5, x4)
        x = self.up22(x, x3)
        x = self.up32(x, x2)
        x = self.up42(x, x1)
        x = self.unet_2nd_out(x)
        return x
