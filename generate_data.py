# we need to make a bunch of X1 second videos @ X2 FPS 
# ^ answers how many frames should our clips be?
# each one's path and x-position on the track will need to go into data.csv

# we need to go through saving every frame as it's own filename
# e.g. ./summer/0000003.avi # counter = 3
counter = counter + 1
# we will also need to filter out images that aren't "different enough"
# diff_enuff == (abs(np.sum(last_frame_that_met_this_criteria) - np.sum(current_candidate_frame)))
# if diff_enuff:
#   capture.write(np_frame)
#   counter = counter + 1
# we need to keep a counter for how many avi clips we've included thus far
