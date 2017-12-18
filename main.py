import torch.utils.data as data_utils
import argparse
from models import *
from dataset import Data
models_dict = {
    'UNet': UNet
}

def get_args():
    def check_args(args):

        # 任何操作都需指定一个合法的模型
        assert args.model in models_dict

        # 如果执行训练过程，则训练集不为空
        if args.train:
            assert args.train_manifest is not None
        

        # assert args.train 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model to use.')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--load', help='load weights')
    parser.add_argument('--batch_size', help='batch size', default=8)
    parser.add_argument('--max_epoch', help='load model', default=80)
    parser.add_argument('--log_dir', help="directory of logging", default=None)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--predict', action='store_true')

    args = parser.parse_args()
    check_args(args)
    return args

def main(args):
    train_dataset = Data(args.train_manifest)
    train_loader = data_utils.DataLoader(train_dataset, args.batch_size)

    eval_loader = None
    if args.eval_manifest:
        eval_dataset = Data(args.eval_manifest)
        eval_loader = data_utils.DataLoader(eval_dataset, args.batch_size)
    
    net = models_dict[args.model](args)

    if args.train:
        net.train(train_loader, eval_loader)
    elif args.eval:
        net.eval()
    else:
        net.predict()

if __name__ == '__main__':
    args = get_args()
    main(args)

