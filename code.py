# IMPORTANT NOTE:
# This code is adapted from https://github.com/umutsimsekli/sgd_tail_index
# To obtain the dependencies please go to https://github.com/umutsimsekli/sgd_tail_index 

import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import alexnet, fc
from utils import get_id, get_data, accuracy
from utils import get_grads, alpha_estimator, alpha_estimator2
from utils import linear_hinge_loss, get_layerWise_norms



def eval(eval_loader, net, crit, opt, args, test=True):

    net.eval()

    # run over both test and train set    
    total_size = 0
    total_loss = 0
    total_acc = 0

    for x, y in eval_loader:
        # loop over dataset
        x, y = x.to(args.device), y.to(args.device)
        opt.zero_grad()
        out = net(x)

        loss = crit(out, y)
        prec = accuracy(out, y)
        bs = x.size(0)

        total_size += int(bs)
        total_loss += float(loss) * bs
        total_acc += float(prec) * bs

    total_acc /= total_size
        
    return total_acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--exit_set_a', default=100, type=float)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_eval', default=100, type=int,
        help='must be equal to training batch size')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--dataset', default='mnist', type=str,
        help='mnist | cifar10 | cifar100')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='fc', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,
        help='NLL | linear_hinge')
    parser.add_argument('--scale', default=64, type=int,
        help='scale of the number of convolutional filters')
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--width', default=100, type=int, 
        help='width of fully connected layers')
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--schedule', action='store_true', default=False)
    parser.add_argument('--preprocess', action='store_true', default=False)
    parser.add_argument('--lr_schedule', action='store_true', default=False)
    args = parser.parse_args()

    # initial setup
    args.path = get_id(args.path)
    if args.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    
    print(args)

    # training setup
    train_loader, test_loader_eval, train_loader_eval, num_classes = get_data(args)

    if args.model == 'fc':
        if args.pca:
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes, input_dim=20*20).to(args.device)
        elif args.dataset == 'mnist':
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes).to(args.device)
        elif args.dataset == 'cifar10':
            net = fc(width=args.width, depth=args.depth, num_classes=num_classes, input_dim=3*32*32).to(args.device)
    elif args.model == 'alexnet':
        net = alexnet(ch=args.scale, num_classes=num_classes).to(args.device)

    print(net)
    
    opt = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.mom,
        weight_decay=args.wd
        )

    if args.lr_schedule:
        milestone = int(args.iterations / 3)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, 
            milestones=[milestone, 2*milestone],
            gamma=0.5)
    
    if args.criterion == 'NLL':
        crit = nn.CrossEntropyLoss().to(args.device)
    elif args.criterion == 'linear_hinge':
        crit = linear_hinge_loss #.to(args.device)
    
    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)
    
    STOP = False

    is_converged = False
    tr_acc = 0
    for i, (x, y) in enumerate(circ_train_loader):

        if (not is_converged):
            if i % 2000 == 0:
                # first record is at the initial point
                tr_acc = eval(train_loader_eval, net, crit, opt, args, test=False)
                print('Acc is', tr_acc)
                

        if (not is_converged):
            if(tr_acc > 90):
                is_converged = True
                weights_optim = torch.Tensor()
                for p in net.parameters():
                    weights_optim = torch.cat((weights_optim, p.flatten()))


        if is_converged:

            weights = torch.Tensor()
            for p in net.parameters():
                weights = torch.cat((weights, p.flatten()))
            diff = weights_optim - weights  
            diff_norm = diff.norm()
            if(diff_norm > args.exit_set_a):
                filename = 'out/W{}_D{}_L{:4.4f}_E{:4.4f}.txt'.format(args.width, args.depth, args.lr, args.exit_set_a)
                with open(filename, 'w') as f:
                    f.write('{}\n'.format(i * args.lr))
                print(i*args.lr)  
                break              
            

        net.train()
        
        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        out = net(x)
        acc_cur = accuracy(out, y)
        loss = crit(out, y)

        # calculate the gradients
        loss.backward()

        # take the step
        opt.step()

        if args.lr_schedule:
            scheduler.step(i)

        if i > args.iterations:
            STOP = True

        if STOP:
            print('eval time {}'.format(i))
            break

    
