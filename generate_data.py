# we need to make a bunch of X1 second videos @ X2 FPS 
# ^ answers how many frames should our clips be?
# each one's path and x-position on the track will need to go into data.csv

# we need to go through saving every frame as it's own filename
# e.g. ./summer/0000003.avi # counter = 3
# we will also need to filter out images that aren't "different enough"
# cutoff_hyperpm = 100
# diff_enuff == (abs(np.sum(last_frame_that_met_this_criteria) - np.sum(current_candidate_frame)) > cutoff_hyperpm)
# if diff_enuff:
#   capture.write(np_frame)
#   counter = counter + 1
# we need to keep a counter for how many avi clips we've added so far
# we can do the differencing using just one clip (e.g. winter)

# a sample datapoint row in our data.csv file looks like:
# (relative_path, 1-dim position label)
# ("./winter/0000069.avi", 0.04)
# ^ before processing the label would be some positive integer count

# the label will be inferred using how many
# clips we end up adding to the dataset
# so we can save each filename and video as 
# we make it to save on RAM/memory usage. 
# then at the end when we have the final 
# clip count, we divide our list of time_steps
# by our total number of clips to get our labels...
# this would require keeping around the filenames
# in a parallel list until the csv is made at the very end

# we know whatever clip we make (e.g. summer), we get 3 more for free (same time points for fall, spring, winter)

def load_video(self, fname):
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

if __name__ == "__main__":
  fname = "./summer.webm"
  load_video(fname)
