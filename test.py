# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:42:48 2025

@author: marjan
"""

import os
import time
import numpy as np
import torch
from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from lib import slowfastnet
from tensorboardX import SummaryWriter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test(model, test_dataloader, criterion, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % params['display'] == 0:
                print('----test----')
                print_string = 'Step: [{0}/{1}]'.format(step + 1, len(test_dataloader))
                print(print_string)
                print_string = 'batch time: {batch_time:.3f}'.format(batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)

    writer.add_scalar('test_loss', losses.avg)
    writer.add_scalar('test_top1_acc', top1.avg)
    writer.add_scalar('test_top5_acc', top5.avg)

def main():
    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    # Only loading the test dataset here
    test_dataloader = \
        DataLoader(
            VideoDataset(params['dataset'], mode='test', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
            batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    print("Loading model")
    model = slowfastnet.resnet50(class_num=params['num_classes'])

    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("Loading pretrained model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-GPU

    criterion = nn.CrossEntropyLoss().cuda()

    # Now only testing, so no optimizer or scheduler is needed
    # Removed training-related components (optimizer, scheduler, etc.)

    # Test the model on the test dataset
    test(model, test_dataloader, criterion, writer)

    writer.close()
    

if __name__ == '__main__':
    main()
