import requests
from ConsumerClass import Consumer
import random
from constants import Constants
from ultralytics import YOLO
import cv2
import torch
import json
import pprint
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import hashlib
import os
from datetime import datetime



class SpacePointSearch(Consumer):
    def __init__(
        self,
        name="Space Point Search",
        id=None,
        queuename=None,
        rabbitmqusername="default",
        rabbitmqpassword="default",
        serverurl=Constants.DEFAULT_SERVER_URL,
    ):
        super().__init__(
            name,
            id,
            queuename,
            rabbitmqusername,
            rabbitmqpassword,
            serverurl,
        )
        self.logic = self.logicfunction
        self.testframeid = 16
        self.processable_columns = [
            "ballz",
        ]
        # Initialize depth estimator (called once during initialization)
        self.depth_estimator = DepthEstimator(
            model_size="base",      # High accuracy
            depth_scale=1499.0,
            use_cache=True           # Fast repeated queries
        )

        self.joinserver()

    def getballcoordinates(self, startframeid, endframeid, videoid):
        returnmap = dict()
        missingframes = []
        for frameid in range(startframeid, endframeid + 1):
            response = requests.post(
                f"{self.server}/checkandreturn",
                json={
                    "frameid": frameid,
                    "columns": self.ballcoordinatescolumns,
                    "videoid": videoid,
                },
            )
            data = response.json()
            if response.status_code == 404 or not data:
                missingframes.append(frameid)
                continue
            if response.status_code == 200:
                for column in self.ballcoordinatescolumns:
                    if column not in data:
                        data[column] = None
                returnmap[frameid] = data
            else:
                raise Exception(
                    f"Failed to get ball coordinates for frame {frameid}: {response.json()}"
                )

        return (
            (True, returnmap)
            if not missingframes
            else (False, self.groupframesintoranges(missingframes))
        )

    def getzpointfromxandy(self, annotated_image_path, xcoordinate, ycoordinate):
        """
        Get Z (depth) coordinate for given X, Y pixel coordinates.
        Uses high-accuracy Depth Anything V2 model.
        
        Args:
            annotated_image_path: Path to the image file
            xcoordinate: X pixel coordinate (can be fractional)
            ycoordinate: Y pixel coordinate (can be fractional)
            
        Returns:
            z_coordinate: Depth value at (x, y)
        """
        try:
            z = self.depth_estimator.getzpointfromxandy(
                annotated_image_path, 
                xcoordinate, 
                ycoordinate
            )
            return z
        except Exception as e:
            print(f"Error getting Z coordinate: {e}")
            return None

    def save_nth_frame(self, video_path, n, output_dir=None, prefix="frame"):
        """
        Save the nth frame from a video to an image file. Filename includes a datetime.

        Args:
            video_path (str): Path to the video file.
            n (int): Zero-based frame index to save.
            output_dir (str|None): Directory to save the image. If None, uses video's directory.
            prefix (str): Prefix for the output filename.

        Returns:
            (bool, str|None): (True, saved_image_path) on success, (False, None) on failure.
        """

        if output_dir is None:
            output_dir = os.path.dirname(video_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (False, None)

        try:
            frame_index = int(n)
            if frame_index < 0:
                return (False, None)

            # Seek to the requested frame and read
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret or frame is None:
                return (False, None)

            # Build filename with datetime (UTC) including milliseconds
            dt = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}_frame{frame_index}_{dt}.jpg"
            save_path = os.path.join(output_dir, filename)

            ok = cv2.imwrite(save_path, frame)
            return (True, save_path) if ok else (False, None)
        finally:
            cap.release()

    def annotate_coordinates_on_image_and_save(self, image_path, coordinates):
        # Dummy implementation for annotating coordinates on an image
        # In a real scenario, this would involve using OpenCV or similar to draw on the image
        annotated_image_path = "/path/to/annotated/image.jpg"  # Placeholder path
        return (True, annotated_image_path)

    def logicfunction(self, messagebody):
        videopath = (
            requests.get(
                f"{self.server}/get-video-path-against-id",
                params={"videoId": messagebody["videoid"]},
            )
            .json()
            .get("videoPath", Constants.DEFAULT_VIDEO_PATH)
        )
        startframeid = messagebody.get("startframeid", 0)
        endframeid = messagebody.get("endframeid", 1000)

        endframeimagepath = "/path/to/end/frame/image.jpg"  # Placeholder path

        ball_coordinates_status, ball_coordinates_data = self.getballcoordinates(
            startframeid, endframeid, messagebody["videoid"]
        )

        if not ball_coordinates_status:
            print(f"Missing ball coordinates for frames: {ball_coordinates_data}")
            # TODO: Check if framestart and end being the same causes any issues downstream
            for missingframestart, missingframeend in ball_coordinates_data:
                self.placerequest(
                    self.ballcoordinatescolumns,
                    messagebody["requestid"],
                    missingframestart,
                    missingframeend,
                    videoid=messagebody["videoid"],
                )

            return False

        operation_status, annotated_image_path = (
            self.annotate_coordinates_on_image_and_save(
                endframeimagepath, ball_coordinates_data
            )
        )

        finalresult = dict()
        for frameid, coords in ball_coordinates_data.items():
            xcoordinate = coords.get("ballx", None)
            ycoordinate = coords.get("bally", None)
            zcoordinate = (
                self.getzpointfromxandy(annotated_image_path, xcoordinate, ycoordinate)
                if xcoordinate is not None and ycoordinate is not None
                else None
            )
            finalresult[frameid] = {"ballz": zcoordinate}

        self.saveresult(finalresult, messagebody["videoid"])

        return True

    def saveresult(self, ball_markup, videoId):
        print("Executing saveresult.... for ", videoId)

        for frameid, coords in ball_markup.items():
            for column, value in coords.items():
                response = requests.post(
                    f"{self.server}/updatecolumn",
                    json={
                        "frameid": int(frameid),
                        "column": column,
                        "value": float(value),  # Convert numpy float32 to Python float
                        "videoid": videoId,
                    },
                )
                if response.status_code == 200:
                    print(f"Updated frame {frameid}, column {column} successfully.")
                else:
                    print(
                        f"Failed to update frame {frameid}, column {column}: {response.json()}"
                    )

        return True

class DepthEstimator:
    """
    High-performance depth estimation class with caching for fast repeated queries.
    Uses Depth Anything V2 for accurate depth estimation.
    """
    
    def __init__(self, model_size="base", depth_scale=1499.0, use_cache=True):
        """Initialize the DepthEstimator."""
        self.depth_scale = depth_scale
        self.use_cache = use_cache
        self.cache = {}
        
        model_names = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf"
        }
        
        self.model_name = model_names.get(model_size.lower(), model_names["base"])
        
        print(f"Loading Depth Anything V2 ({model_size}) model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        
        if self.device.type == "cuda":
            self.model = self.model.half()
            torch.backends.cudnn.benchmark = True
        
        print(f"Model loaded on {self.device}")
    
    def _get_image_hash(self, image_path):
        """Generate hash of image for caching."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _estimate_depth_map(self, image_path):
        """Estimate depth map from image with caching."""
        if self.use_cache:
            img_hash = self._get_image_hash(image_path)
            if img_hash in self.cache:
                return self.cache[img_hash]
        
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        
        inputs = self.processor(images=img, return_tensors="pt")
        
        if self.device.type == "cuda":
            inputs = {k: v.half().to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().float().numpy()
        
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth_map = (depth * self.depth_scale).astype(np.float32)
        
        if self.use_cache:
            img_hash = self._get_image_hash(image_path)
            self.cache[img_hash] = (depth_map, (w, h))
        
        return depth_map, (w, h)
    
    def _bilinear_interpolate(self, depth_map, x, y):
        """Fast bilinear interpolation for sub-pixel accuracy."""
        h, w = depth_map.shape
        
        x = np.clip(x, 0, w - 1.001)
        y = np.clip(y, 0, h - 1.001)
        
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        fx, fy = x - x0, y - y0
        
        z00 = depth_map[y0, x0]
        z01 = depth_map[y0, x1]
        z10 = depth_map[y1, x0]
        z11 = depth_map[y1, x1]
        
        z = (z00 * (1 - fx) * (1 - fy) +
             z01 * fx * (1 - fy) +
             z10 * (1 - fx) * fy +
             z11 * fx * fy)
        
        return float(z)
    
    def getzpointfromxandy(self, image_path, x, y):
        """Get Z (depth) coordinate for given X, Y pixel coordinates."""
        depth_map, img_size = self._estimate_depth_map(image_path)
        
        w, h = img_size
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(
                f"Coordinates ({x}, {y}) out of bounds. "
                f"Image size: {w}x{h}"
            )
        
        z_coordinate = self._bilinear_interpolate(depth_map, x, y)
        
        return z_coordinate


if __name__ == "__main__":
    c1 = SpacePointSearch(
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
        id="space-point-searchs",
    )
    c1.threadstart()
