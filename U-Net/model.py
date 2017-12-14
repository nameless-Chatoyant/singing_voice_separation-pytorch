import torch
import torch.nn as nn
import torch.nn.functional as F
from cfgs.config import cfg

class UNet(nn.Module):
    def __init__(self):
        self.convs = []
        self.deconvs = []
        for i in range(len(cfg.encoder.ch_out)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = cfg.encoder.ch_in[i],
                                out_channels = cfg.encoder.ch_out[i], 
                                kernel_size = cfg.encoder.kernel_size,
                                stride = cfg.encoder.stride
                                ),
                    nn.BatchNorm2d(cfg.encoder.ch_out[i]),
                    nn.LeakyReLU(cfg.encoder.leakiness),
                )
            )
        for i in range(len(cfg.decoder.ch_out)):
            self.deconvs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = cfg.decoder.ch_in[i],
                        out_channels = cfg.decoder.ch_out[i],
                        kernel_size = cfg.decoder.kernel_size,
                        stride = cfg.decoder.stride,
                    ),
                    nn.BatchNorm2d(cfg.decoder.ch_out[i]),
                    nn.ReLU()
                )
            )
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        for layer_idx, deconv in enumerate(self.deconvs):
            x = deconv(x)
            if layer_idx < 3:
                x = F.Dropout2d(x, p = 0.5)

if __name__ == '__main__':
    unet = UNet()
    import numpy as np
    from torch.autograd import Variable
    print(unet.forward(Variable(torch.Tensor(np.ones((1,512,128,1))))))   