from model import UNet

net = UNet()
if cuda:
    net.cuda()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size', help='load model', default=8)
    parser.add_argument('--log_dir', help="directory of logging", default=None)
    args = parser.parse_args()

    for epoch in()
    net.train(epoch)