import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):

    def __init__(self, directory, mode='train', clip_len=8, frame_sample_rate=1):
        # @TODO make train mode grab from 3 folders and test from the remaining (i.e. summer)
        if mode == 'train' or mode == 'training':
            spring = os.path.join(directory, "spring.webm")
            fall = os.path.join(directory, "fall.mp4")
            winter = os.path.join(directory, "winter.webm")
            folders = [spring, fall, winter]
        else:
            summer = os.path.join(directory, "summer.webm")
            folders = [summer]
        self.clip_len = clip_len

        self.short_side = [128, 160]
        self.crop_size = 112
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode

        self.fnames, labels = [], []
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rpath = row["relative_path"]
                label = row["label"]
                for folder in folders:
                    self.fnames.append(os.path.join(folder, rpath))
                    labels.append(label)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        # notice loadvideo returns our buffer (i.e. 4d clip tensor)
        # it is taking a filename (i.e. fname) from our dataset
        # and loading it into a 4D pytorch tensor.
        buffer = self.loadvideo(self.fnames[index])

        # I think we're skipping clips that are too short
        while buffer.shape[0]<self.clip_len+2 :
            index = np.random.randint(self.__len__())
            buffer = self.loadvideo(self.fnames[index])

        if self.mode == 'train' or self.mode == 'training':
            # here's some data-aug
            buffer = self.randomflip(buffer)
        # self.crop will crop X,Y,& time
        # where you begin cropping the subset is a random number,
        # hence giving us some more data augmentation!
        # time is cropped down to the desired clip-length
        # before this cropping, the buffer could have 300 vid frames
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        # our input to the network and label
        # 2d resnet's input would be 3D.
        # 3d resnet's is a clip that is 4d.
        # buffer == clip == sequence of frames
        return buffer, self.labels[index]

    def to_tensor(self, buffer):
        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        # Now it is RGB-dim first, time/Z, Y, and X as the last dimension.
        # This is why we index the 2nd-dim (counting from 0), 
        # keeping in mind that the first dim is the batch during training.
        return buffer.transpose((3, 0, 1, 2))

    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        # cv2 is opencv2, a fast python library for doing image and video processing fxs
        capture = cv2.VideoCapture(fname)
        # notice the capture has statistics about its video
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    #z
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))    #X
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  #Y

        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count - 1
 
        # Z-dim (i.e. time), Y-dim, X-dim, RGB-dim
        buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))

        count = 0
        # doesn't have to be initialized technically
        retaining = True

        # read in each frame, (potentially) one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            # this is how you get each from of a video using Open-CV2
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            # the first var from read() is whether the video is empty/done
            if retaining is False or count > end_idx:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # will resize frames if not already final size

            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))
            buffer[count] = frame
            count += 1
        capture.release() # we're done with the video object from opencv-2
        return buffer
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]]))/128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':

    datapath = r'D:\SlowFastNN\UCF-101\archive'
    train_dataloader = \
        DataLoader( VideoDataset(datapath, mode='train'), batch_size=10, shuffle=True, num_workers=0)
    for step, (buffer, label) in enumerate(train_dataloader):
        print("label: ", label)
