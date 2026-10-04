import json
from pathlib import Path
import cv2
import os

def save_json(data: dict, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_frames_from_video(video_path: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    print(f"Extracted {frame_idx} frames to {output_dir}")
    return frame_idx
