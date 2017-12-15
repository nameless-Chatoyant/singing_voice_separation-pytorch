from models import *

import torch.utils.data as data_utils

from dataset import Data

batch_size = 8
train_dataset = Data('train')
test_dataset = Data('train')
train_loader = data_utils.DataLoader(train_dataset, batch_size)
test_loader = data_utils.DataLoader(test_dataset, batch_size)


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




def train(model, train_iter, dev_iter, loss_func, args):
    """
    # Arguments
        model:
        train_iter:
        dev_iter:
        loss_func:
        args:
    """
    if args.cuda:
        model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    steps = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                
                output = model(feature)

                loss = loss_func(output, target)
                loss.backward()
                optimizer.step()

                steps += 1

                if steps % args.test_interval == 0:
                    eval()
                if steps % args.save_interval == 0:
                    torch.save(model, save_path)
    
def eval(model, data_iter, args):
    model.eval()
    
    for batch in data_iter:
        feature, target = batch.text, batch.label
        output = model(feature)
        loss = loss_func(output, feature)
        
    
    model.train()


    np.arange(*[0.5, 1.0], 5)