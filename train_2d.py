import os
import time
import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from lib.dataset import VideoDataset
from torchvision import datasets, models

cudnn.benchmark = False

image_datasets = {}
# directory, mode='train', frame_sample_rate=1, dim=3
image_datasets['train'] = VideoDataset('./', mode='train', dim=2)
image_datasets['val'] = VideoDataset('./', mode='test', dim=2)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, 
                                              num_workers=8, pin_memory=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

assert torch.cuda.is_available(), "GPU not found!"
device = torch.device("cuda:0")

def imshow(inp, title=None):
    """Display image for Tensor."""
    # (C,H,W) -> (H,W,C) *not sure about order of H & W
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([128, 128, 128])
    std = np.array([128, 128, 128])
    # this undoes the normalization in dataset.py
    # making the image look as we typically expect
    inp = std * inp + mean
    # assumes 8-bit encoding
    # which allows pixel values of 0-255
    # plt expects everything between 0 & 1 though
    inp = inp / 255
    inp = np.clip(inp, 0, 1)
    # since the final dimension is the RGB, this reverses the order
    # most software libraries use one of the other, so try this 
    # trick if you're images contain the right shape and brightness
    # but the colors seem weird / inverted. 
    plt.imshow(inp[...,::-1]) # changes from the red green blue to blue green red
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit for the plots

# Get a batch of training data
inputs, _x = next(iter(dataloaders['train']))
inputs = inputs[:8,...]
out = torchvision.utils.make_grid(inputs)
imshow(out, title="Verify the images loaded correctly!")
plt.show() 

def train_model(model, criterion, optimizer, scheduler, num_epochs=1): # scheduler change the learning rate dynamically
    start_time = time.time()
    best_loss = 1e8
    best_model_params_path = "./best_model_params.pt"

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0 # accumiliated loss

            # Iterate over data.
            for inputs, x in dataloaders[phase]: # in each batch
                inputs = inputs.to(device)
                labels = x.to(device)
                #labels = torch.squeeze(labels, -1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(1), labels)
                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            if phase == 'train' and scheduler != None:
                scheduler.step() # dynamically decrease the learning rate

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), best_model_params_path)
                print("new best!")

        print("")

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model

def visualize_model(model, num_images=16):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for inputs, x in dataloaders['val']:
            inputs = inputs.to(device)
            labels = x.to(device)
            outputs = model(inputs)
        
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//4, 4, images_so_far)
                ax.axis('off')
                o_x = float(outputs[j].cpu())
                gt_x = float(labels[j].cpu())
                title = f'Predicted: ({o_x:.3f}) \nGT: ({gt_x:.3f})'
                ax.set_title(title, fontsize=8)
                imshow(inputs.cpu().data[j])
        
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
                
        model.train(mode=was_training)

if __name__ == "__main__":
    # get the model definition and weights for a resnet18
    # pre-trained on the full imagenet (classes=1,000!) dataset
    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    # print(model_conv) will let you see the layers of the network
    num_ftrs = model_conv.fc.in_features # last layer of resenet is fc. we need to know the num input to that layer. The num output is the same
    # chopped off the head and replaced with our 1 output neuron
    model_conv.fc = nn.Sequential(
         nn.Linear(num_ftrs, 1),
         nn.Sigmoid(),
    )
    
    model_conv = model_conv.to(device)
    
    criterion = nn.MSELoss()
    # optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.01, momentum=0.9)
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=1e-3)
    lr_schedule = None # lr_scheduler.StepLR(optimizer_conv, step_size=50, gamma=0.1)
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             lr_schedule, num_epochs=250)
    
    # model_conv.load_state_dict(torch.load("./best_model_params.pt"))
    visualize_model(model_conv)
    #make_movie(model_conv) # see original code...
    plt.show()
