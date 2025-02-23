import cv2
import numpy as np
import os
import csv

# Parameters
SHRINK_FACTOR = 50
CLIP_DURATION = 24 * SHRINK_FACTOR          # seconds per clip (X1 seconds)
FPS = 1                                     # frames per second (X2 FPS)
FRAME_DIFF_CUTOFF = 1000                    # cutoff hyperparameter for frame difference
SHORT_SIDE_RANGE = (500, 700)               # Example range for resizing the shorter side

# Input video files per season
INPUT_VIDEOS = {
    #"summer": "./summer.mp4",
    "fall":   "./fall.mp4",
    "spring": "./spring.webm",
    "winter": "./winter.webm"
}

# Create output directories for each season
for season in INPUT_VIDEOS.keys():
    output_dir = f"./{season}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
CSV_FILENAME = "data.csv"  # single CSV logging relative path and label (clip index)

def get_resize_dims(frame_width, frame_height):
    # Determine new dimensions while keeping aspect ratio based on a random choice in SHORT_SIDE_RANGE.
    if frame_height < frame_width:
        resize_height = np.random.randint(SHORT_SIDE_RANGE[0], SHORT_SIDE_RANGE[1] + 1)
        resize_width = int(float(resize_height) / frame_height * frame_width)
    else:
        resize_width = np.random.randint(SHORT_SIDE_RANGE[0], SHORT_SIDE_RANGE[1] + 1)
        resize_height = int(float(resize_width) / frame_width * frame_height)
    return resize_width, resize_height

def save_clip(frames_dict, clip_index, fps=FPS):
    # Save one clip per season, using the provided frames dictionary.
    # The frames_dict should have season as key and list of frames as value.
    for season, frames in frames_dict.items():
        if not frames:
            continue
        # Only save every SHRINK_FACTOR-th frame
        sampled_frames = frames[::SHRINK_FACTOR]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        height, width, _ = sampled_frames[0].shape
        # Adjust fps according to SHRINK_FACTOR if needed
        new_fps = max(1, fps // SHRINK_FACTOR)
        filename = os.path.join(f"./{season}", f"{clip_index:07d}.avi")
        out = cv2.VideoWriter(filename, fourcc, new_fps, (width, height))
        for frame in sampled_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

def process_videos(input_videos):
    # Open VideoCapture for all videos (assumed aligned in time)
    caps = {}
    for season, path in input_videos.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file for {season}: {path}")
            return
        caps[season] = cap

    # Decide on a common resize dimension using the winter video (assumed all have same dimensions)
    frame_width = int(caps["winter"].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps["winter"].get(cv2.CAP_PROP_FRAME_HEIGHT))
    resize_width, resize_height = get_resize_dims(frame_width, frame_height)
    
    clip_frames = CLIP_DURATION * FPS  # Number of frames per clip (accepted frames)
    accepted_clip_count = 0
    # Store filenames and corresponding raw clip counts (for labels)
    filenames = []
    time_steps = []  # one entry per clip (index)
    
    # For each season, maintain a current clip buffer and last accepted frame (for frame differencing)
    current_clips = {season: [] for season in input_videos.keys()}
    last_accepted = {season: None for season in input_videos.keys()}
    
    frame_idx = 0
    start_skip_amount = 6000  # skip over black or initial frames (if needed)
    skips = 0

    while True:
        # Read one frame from each video
        frames = {}
        end_reached = False
        for season, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                end_reached = True
                break
            # For the first few frames, skip if needed
            if skips < start_skip_amount:
                frames[season] = None
                continue
            # Convert to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (frame.shape[1] != resize_width) or (frame.shape[0] != resize_height):
                frame = cv2.resize(frame, (resize_width, resize_height))
            frames[season] = frame
        if end_reached:
            break
        
        if skips < start_skip_amount:
            skips += 1
            frame_idx += 1
            continue
        
        # Process each season's frame for frame differencing
        for season, frame in frames.items():
            if frame is None:
                continue
            if len(current_clips[season]) == 0:
                # Start a new clip: always accept the first frame.
                current_clips[season].append(frame)
                last_accepted[season] = frame
            else:
                diff = abs(np.sum(last_accepted[season].astype("float32")) - np.sum(frame.astype("float32")))
                # Debug print can be uncommented if needed:
                print(f"{season} Frame {frame_idx} diff: {diff}")
                if diff > FRAME_DIFF_CUTOFF:
                    current_clips[season].append(frame)
                    last_accepted[season] = frame

        # Check if any one season’s clip buffer reached the required length.
        # (Since clips are in parallel, they should all roughly reach the threshold together.)
        if all(len(clip) >= clip_frames for clip in current_clips.values()):
            # Save out clips for all seasons with the same clip index.
            save_clip(current_clips, accepted_clip_count, fps=FPS)
            # Record filename of one season (say winter) as the reference; label applies equally.
            filenames.append(os.path.join("winter", f"{accepted_clip_count:07d}.avi"))
            time_steps.append(accepted_clip_count)
            accepted_clip_count += 1
            # Reset clip buffers and last accepted frames for all seasons
            current_clips = {season: [] for season in input_videos.keys()}
            last_accepted = {season: None for season in input_videos.keys()}
        
        frame_idx += 1

    # Release all captures
    for cap in caps.values():
        cap.release()

    # Normalize the time_steps to obtain labels between 0 and 1.
    if accepted_clip_count > 1:
        normalized_labels = [ts / (accepted_clip_count - 1) for ts in time_steps]
    else:
        normalized_labels = [0 for _ in time_steps]

    # Write CSV file with (relative_path, normalized label)
    with open(CSV_FILENAME, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["relative_path", "label"])
        for fn, label in zip(filenames, normalized_labels):
            writer.writerow([fn, label])
    print(f"Processed {accepted_clip_count} clips. CSV saved as {CSV_FILENAME}.")

if __name__ == "__main__":
    process_videos(INPUT_VIDEOS)
