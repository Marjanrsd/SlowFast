# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:16:55 2025

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
'''
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
    print(f'batch size is {batch_size}')
    # in each batch for each sample return top 1 or 5 acc and the indices
    _, pred = output.topk(maxk, 1, True, True) # the first True means that the function will return the values sorted in descending order - second True:indices
    print(f'pred {pred}')
    print(f'pred shape is {pred.shape}')
    pred = pred.t() # transpose
    print(f'pred transpose {pred}')
    print(f'pred transpose shape is {pred.shape}')
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print(f'correct {correct}')
    print(f'correct shape is {correct.shape}')

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        print(f"correct_{k} is {correct_k}")
        res.append(correct_k.mul_(100.0 / batch_size))
  
    return res
'''
'''
def train(model, train_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, (inputs, labels) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad() # reset gradient for each batch
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (step+1) % params['display'] == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step+1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)
    writer.add_scalar('train_loss_epoch', losses.avg, epoch)
    writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)
'''
'''
def validation(model, val_dataloader, epoch, criterion, optimizer, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
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
                print('----validation----')
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)
    writer.add_scalar('val_loss_epoch', losses.avg, epoch)
    writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
    writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)


def main():
    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    logdir = os.path.join(params['log'], cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(log_dir=logdir)

    print("Loading dataset")
    #train_dataloader = \
        #DataLoader(
           # VideoDataset(params['dataset'], mode='train', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
            #batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            VideoDataset(params['dataset'], mode='val', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
            batch_size=params['batch_size'], shuffle=False, num_workers=params['num_workers'])

    print("load model")
    model = slowfastnet.resnet50(class_num=params['num_classes'])
    
    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("load pretrain model")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model = model.cuda(params['gpu'][0])
    model = nn.DataParallel(model, device_ids=params['gpu'])  # multi-Gpu

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['step'], gamma=0.1)

    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for epoch in range(params['epoch_num']):
        #train(model, train_dataloader, epoch, criterion, optimizer, writer)
        #if epoch % 2== 0:
        validation(model, val_dataloader, epoch, criterion, optimizer, writer)
        scheduler.step()
        #if epoch % 1 == 0:
            #checkpoint = os.path.join(model_save_dir,
                                     # "clip_len_" + str(params['clip_len']) + "frame_sample_rate_" +str(params['frame_sample_rate'])+ "_checkpoint_" + str(epoch) + ".pth.tar")
            #torch.save(model.module.state_dict(), checkpoint)

    writer.close

if __name__ == '__main__':
    main()
'''
# get the time
def test_inference_speed(test_video_path, batch_size=1):
    """
    Function to test how fast the model can process video data.
    """
        # Load model
    print("Loading model...")
    model = slowfastnet.resnet50(class_num=params['num_classes'])
    
    if params['pretrained'] is not None:
        pretrained_dict = torch.load(params['pretrained'], map_location='cpu')
        try:
            model_dict = model.module.state_dict()
        except AttributeError:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print("Loading pretrained model...")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    #  batch size is set to the number of clips per batch
    # each video contains multiple clips
    # each clip contains multiple frames
    # Clip length (clip_len) refers to the number of frames in each clip
    # Move the model to GPU if available
    model = model.cuda(params['gpu'][0])
    model = torch.nn.DataParallel(model, device_ids=params['gpu'])  # multi-GPU support
    model.eval()  # Set model to evaluation mode
    video_dataset = VideoDataset(
        test_video_path, 
        mode='test', 
        clip_len=params['clip_len'], 
        frame_sample_rate=params['frame_sample_rate']
    )
    test_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, num_workers=params['num_workers'])

    total_time = 0
    num_batches = len(test_dataloader)
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, _ in test_dataloader:
            # Move data to GPU if available
            inputs = inputs.cuda()

            # Start the timer
            start_time = time.time()
            
            # Run the model on the input data
            outputs = model(inputs)
            
            # Stop the timer
            end_time = time.time()
            print(f'time took to preocess one batch: {end_time - start_time}')

            # Compute the time for this batch
            total_time += (end_time - start_time)
            
            
    # Calculate average inference time per video (or per batch)
    avg_time = total_time / num_batches
    print(f"Average inference time per batch: {avg_time:.4f} seconds")
    
    # You can also calculate FPS (frames per second) 
    fps = 1 / avg_time
    print(f"Frames per second (FPS): {fps:.2f} FPS")

if __name__ == '__main__':
 
    test_inference_speed(r"D:\SlowFastNN\test_for_time", batch_size=1)
