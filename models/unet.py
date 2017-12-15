import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, ceil

def same_padding_conv(x, conv):
    dim = len(x.size())
    if dim == 4:
        b, c, h, w = x.size()
    elif dim == 5:
        b, t, c, h, w = x.size()
    else:
        raise NotImplementedError()

    if isinstance(conv, nn.Conv2d):
        padding = ((w // conv.stride[0] - 1) * conv.stride[0] + conv.kernel_size[0] - w)
        padding_l = floor(padding / 2)
        padding_r = ceil(padding / 2)
        padding = ((h // conv.stride[1] - 1) * conv.stride[1] + conv.kernel_size[1] - h)
        padding_t = floor(padding / 2)
        padding_b = ceil(padding / 2)
        x = F.pad(x, pad = (padding_l,padding_r,padding_t,padding_b))
        x = conv(x)
    elif isinstance(conv, nn.ConvTranspose2d):
        padding = ((w - 1) * conv.stride + conv.kernel_size[0] - w * conv.stride[0])
        padding_l = floor(padding / 2)
        padding_r = ceil(padding / 2)
        padding = ((h - 1) * conv.stride + conv.kernel_size[1] - h * conv.stride[1])
        padding_t = floor(padding / 2)
        padding_b = ceil(padding / 2)
        x = conv(x)
        x = x[:,:,padding_t:-padding_b,padding_l:-padding_r]
    else:
        raise NotImplementedError()
    return x

class config:
    class encoder:
        leakiness = 0.2
        ch_in = [1, 16, 32, 64, 128, 256]
        ch_out = [16, 32, 64, 128, 256, 512]
        kernel_size = (5, 5)
        stride = 2
    class decoder:
        ch_in = [512, 512, 256, 128, 64, 32]
        ch_out = [256, 128, 64, 32, 16]
        kernel_size = (5, 5)
        stride = 2


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.convs = []
        self.deconvs = []
        self.kernel_size = config.encoder.kernel_size
        self.stride = config.encoder.stride
        for i in range(len(config.encoder.ch_out)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels = config.encoder.ch_in[i],
                        out_channels = config.encoder.ch_out[i], 
                        kernel_size = self.kernel_size,
                        stride = self.stride
                    ),
                    nn.BatchNorm2d(config.encoder.ch_out[i]),
                    nn.LeakyReLU(config.encoder.leakiness),
                )
            )
        for i in range(len(config.decoder.ch_out)):
            self.deconvs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = config.decoder.ch_in[i],
                        out_channels = config.decoder.ch_out[i],
                        kernel_size = config.decoder.kernel_size,
                        stride = config.decoder.stride
                    ),
                    nn.BatchNorm2d(config.decoder.ch_out[i]),
                    nn.ReLU()
                )
            )
    def forward(self, x):
        conv_output = []
        skip_connections = []
        for layer_idx, conv in enumerate(self.convs):
            x = same_padding_conv(x, conv)
            if layer_idx != len(self.convs) - 1:
                skip_connections.append(x)
        for layer_idx, deconv in enumerate(self.deconvs):
            x = same_padding_conv(x, deconv)
            if layer_idx < 3:
                x = F.dropout2d(x, p = 0.5)
            x = torch.cat([skip_connections.pop(), x], dim = 1)
        return x

if __name__ == '__main__':
    unet = UNet(config)
    print(unet)

    import numpy as np
    from torch.autograd import Variable
    unet.forward(Variable(torch.Tensor(np.ones((7,1,512,128)))))
