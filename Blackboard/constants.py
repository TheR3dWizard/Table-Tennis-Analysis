import os
from pathlib import Path
from dotenv import load_dotenv

# load .env from project root if present
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

class Constants:
    IMAGE1_PATH = "assets/masked/i3.jpeg"
    IMAGE2_PATH = "assets/masked/i2.jpeg"
    HOUGH_RHO = 1
    # HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD = 80
    HOUGH_MIN_LINE_LENGTH = 200
    HOUGH_MAX_LINE_GAP = 1
    OUTPUT_IMAGE_PATH = "../assets/results/image.jpeg"
    TABLE_VERTEX_DETECTION_WEIGHTS = "../weights/TableDetection.pt"
    BALL_POSITION_DETECTION_WEIGHTS = "../weights/BallDetection.pt"
    DEFAULT_VIDEO_PATH = "../assets/rallies_02.mp4"
    DEFAULT_OUTPUT_FOLDER_PATH = "../assets/output/"
    DEFAULT_FILE_SAVE_PATH = "../storage/"
    DEFAULT_SERVER_URL = "http://localhost:6060"
    RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME", "pw1tt")
    RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "securerabbitmqpassword")
    RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
    RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", 5672)
    POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME", "pw1tt")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "securepostgrespassword")
    POSTGRES_DBNAME = os.getenv("POSTGRES_DBNAME", "blackboard")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", 5432)
    YOLOV8N_WEIGHTS_PATH = os.getenv("YOLOV8N_WEIGHTS_PATH", "../weights/yolov8n.pt")
    YOLO11N_POSE_WEIGHTS_PATH = os.getenv("YOLO11N_POSE_WEIGHTS_PATH", "../weights/yolov11n-pose.pt")
