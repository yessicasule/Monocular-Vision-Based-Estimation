import os
import yaml
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
LANDMARKS_DIR = DATA_DIR / "landmarks"
ANGLES_DIR = DATA_DIR / "angles"
OUTPUTS_DIR = BASE_DIR / "outputs"

DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "default.yaml"

class Config:
    # Default values in case YAML is not found
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    UDP_IP = "127.0.0.1"
    UDP_PORT = 9000
    STREAM_HZ = 30.0
    DEFAULT_FILTER_TYPE = "kalman"
    ANGLE_UNITS = "degrees"
    
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_COMPLEXITY = 1
    SMOOTH_LANDMARKS = True
    
    RECORD_DURATION = 60
    OUTPUT_VIDEO_FPS = 30
    VIDEO_CODEC = 'mp4v'
    
    @classmethod
    def load_from_yaml(cls, yaml_path: Path | str):
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            if not data:
                return
                
            # Map YAML keys to class attributes
            if "camera" in data:
                cam = data["camera"]
                if "id" in cam: cls.CAMERA_ID = cam["id"]
                if "width" in cam: cls.FRAME_WIDTH = cam["width"]
                if "height" in cam: cls.FRAME_HEIGHT = cam["height"]
            if "stream" in data:
                stream = data["stream"]
                if "udp_ip" in stream: cls.UDP_IP = stream["udp_ip"]
                if "udp_port" in stream: cls.UDP_PORT = stream["udp_port"]
                if "hz" in stream: cls.STREAM_HZ = stream["hz"]
            if "filter" in data:
                filt = data["filter"]
                if "default_type" in filt: cls.DEFAULT_FILTER_TYPE = filt["default_type"]
                if "units" in filt: cls.ANGLE_UNITS = filt["units"]
            if "pose" in data:
                pose = data["pose"]
                if "min_detection_confidence" in pose: cls.MIN_DETECTION_CONFIDENCE = pose["min_detection_confidence"]
                if "min_tracking_confidence" in pose: cls.MIN_TRACKING_CONFIDENCE = pose["min_tracking_confidence"]
                if "model_complexity" in pose: cls.MODEL_COMPLEXITY = pose["model_complexity"]
                if "smooth_landmarks" in pose: cls.SMOOTH_LANDMARKS = pose["smooth_landmarks"]
            if "record" in data:
                rec = data["record"]
                if "duration" in rec: cls.RECORD_DURATION = rec["duration"]
                if "fps" in rec: cls.OUTPUT_VIDEO_FPS = rec["fps"]
                if "codec" in rec: cls.VIDEO_CODEC = rec["codec"]
        except Exception as e:
            print(f"Warning: Failed to load config from {yaml_path}: {e}")

    @classmethod
    def ensure_directories(cls):
        for d in [RAW_VIDEOS_DIR, FRAMES_DIR, LANDMARKS_DIR, ANGLES_DIR, OUTPUTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        print("Output directories created/verified")

# Load default config at import time
Config.load_from_yaml(DEFAULT_CONFIG_PATH)

# Module-level alias constants referencing Config to prevent drift/duplication
UDP_IP = Config.UDP_IP
UDP_PORT = Config.UDP_PORT
STREAM_HZ = Config.STREAM_HZ
DEFAULT_FILTER_TYPE = Config.DEFAULT_FILTER_TYPE
ANGLE_UNITS = Config.ANGLE_UNITS
