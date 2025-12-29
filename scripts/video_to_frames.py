import cv2
import os
from tqdm import tqdm

# Path to input video(s)
video_dir = "../data/videos"
# Path to save frames
frames_dir = "../data/frames"

os.makedirs(frames_dir, exist_ok=True)

# Loop over all videos
videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

print(f"Found {len(videos)} videos")

for video_file in tqdm(videos):
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(video_file)[0]

    # Create folder for frames
    save_path = os.path.join(frames_dir, video_name)
    os.makedirs(save_path, exist_ok=True)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame
        frame_filename = os.path.join(save_path, f"{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1

    cap.release()
    print(f"{video_file}: {count} frames saved")

