import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(object):
    def __init__(self, args):
        pass
    def get_model(self):
        pass
    def get_optimizer(self):
        pass
    def loss_func(self, *args):
        pass
    def train(self, train_iter, eval_iter, args):
        model = self.get_model()
        if args.cuda:
            model.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

        steps = 0
        model.train()
        for epoch in range(args.epochs):
            for batch in train_iter:
                feature, target = batch.feature, batch.target
                # feature.
                if args.cuda:
                    feature, target = feature.cuda(), target.cuda()

                optimizer.zero_grad()
                output = model(feature)
                loss = self.loss_func(output, target)
                loss.backward()
                optimizer.step()

                steps += 1
                save_path = 0
                if steps % args.test_interval == 0 and eval_iter:
                    self.eval(eval_iter)
                if steps % args.save_interval == 0:
                    torch.save(model, save_path)
    def eval(self, eval_iter):
        pass