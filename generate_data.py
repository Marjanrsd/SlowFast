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

# the label will be inferred using how many
# clips we end up adding to the dataset
# so we can save each filename and video as 
# we make it to save on RAM/memory usage. 
# then at the end when we have the final 
# clip count, we divide our list of time_steps
# by our total number of clips to get our labels...
# this would require keeping around the filenames
# in a parallel list until the csv is made at the very end
