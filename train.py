from models import *

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

net = UNet(config)
if True:
    net.cuda()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size', help='load model', default=8)
    parser.add_argument('--max_epoch', help='load model', default=80)
    parser.add_argument('--log_dir', help="directory of logging", default=None)
    args = parser.parse_args()

    for epoch in range(args.max_epoch):
        net.train(epoch)