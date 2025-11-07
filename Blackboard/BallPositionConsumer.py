import requests
from ConsumerClass import Consumer
import random
from constants import Constants
from ultralytics import YOLO
import cv2
import json
from tqdm import tqdm


class Ball2DPositionConsumer(Consumer):
    def __init__(
        self,
        name="Ball 2D Position Detection",
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
            "ballx",
            "bally",
        ]
        self.joinserver()

    def computerandomballcoordinates(self):
        self.newprint("Executing computerandomballcoordinates....")
        x, y, z = (
            random.uniform(0, 1000),
            random.uniform(0, 1000),
            random.uniform(0, 1000),
        )

        return {
            "ballx": x,
            "bally": y,
            "ballz": z,
        }

    def run_yolo_model_on_image(
        self, image_path, model_path=Constants.YOLOV8N_WEIGHTS_PATH
    ):
        model = YOLO(model_path)

        img = cv2.imread(image_path)

        results = model(img)

        return results

    def run_yolo_on_video(
        self, model_path, video_path, start_frame=0, end_frame=None, conf=0.25
    ):
        model = YOLO(model_path)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If end_frame not provided, use till last frame
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames

        # Jump to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Calculate frames to process
        frames_to_process = end_frame - start_frame
        
        # Create progress bar
        progress_bar = tqdm(
            total=frames_to_process,
            desc="Processing video frames",
            unit="frame",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}, {rate_fmt}]"
        )

        results_dict = {}
        frame_num = start_frame
        frames_with_ball = 0

        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            preds = model.predict(frame, conf=conf, verbose=False)

            ball_detected = False
            for r in preds:
                boxes = r.boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    results_dict[str(frame_num)] = {
                        "ballx": cx,
                        "bally": cy,
                        "ballvisibility": True,
                    }
                    ball_detected = True

            if ball_detected:
                frames_with_ball += 1
            
            frame_num += 1
            
            # Update progress bar
            current_processed = frame_num - start_frame
            progress_bar.update(1)
            progress_bar.set_postfix({"Ball detected": f"{frames_with_ball}/{current_processed}"})

        progress_bar.close()
        cap.release()

        return results_dict, json.dumps(results_dict, indent=4)

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
        ball_markup, json_output = self.run_yolo_on_video(
            Constants.BALL_POSITION_DETECTION_WEIGHTS,
            videopath,
            startframeid,
            endframeid,
            conf=0.25,
        )

        missingcoordinate = {"ballx": -1, "bally": -1, "ballvisibility": False}

        for i in range(messagebody["startframeid"], messagebody["endframeid"] + 1):
            if str(i) not in ball_markup:
                ball_markup[str(i)] = missingcoordinate

        self.saveresult(ball_markup, messagebody["videoid"], startframeid, endframeid)

        return True

    def saveresult(self, ball_markup, videoId, startframeid, endframeid):
        self.newprint("Executing saveresult.... for ", videoId)

        # Calculate total frames to update
        total_frames = len(ball_markup)
        frames_updated = 0
        
        # Create progress bar with custom format
        progress_bar = tqdm(
            total=total_frames,
            desc="Saving ball positions",
            unit="frame",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for frameid, coords in ball_markup.items():
            frame_success = True
            for column, value in coords.items():
                response = requests.post(
                    f"{self.server}/updatecolumn",
                    json={
                        "frameid": int(frameid),
                        "column": column,
                        "value": (
                            value if value in {True, False} else float(value)
                        ),  # Convert numpy float32 to Python float
                        "videoid": videoId,
                    },
                )
                if response.status_code != 200:
                    frame_success = False
                    self.newprint(
                        f"Failed to update frame {frameid}, column {column}: {response.json()}",
                        event="updatecolumn1",
                        level="error",
                    )
            
            # Update progress bar after processing all columns for this frame
            if frame_success:
                frames_updated += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"Updated": f"{frames_updated}/{total_frames}"})

        progress_bar.close()
        self.newprint(f"Successfully updated {frames_updated}/{total_frames} frames", event="saveresult_complete")

        return True


if __name__ == "__main__":
    c1 = Ball2DPositionConsumer(
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
        id="ball-2d-position-detection",
    )
    c1.threadstart()
