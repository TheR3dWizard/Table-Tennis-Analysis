import requests
from ConsumerClass import Consumer
import random
from constants import Constants
from ultralytics import YOLO
import cv2
import json


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

        results_dict = {}
        frame_num = start_frame

        while frame_num < end_frame:
            self.newprint(f"Processing frame {frame_num}", end="\r")
            ret, frame = cap.read()
            if not ret:
                break

            preds = model.predict(frame, conf=conf, verbose=False)

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

            frame_num += 1

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

        self.saveresult(ball_markup, messagebody["videoid"])

        return True

    def saveresult(self, ball_markup, videoId):
        self.newprint("Executing saveresult.... for ", videoId)

        for frameid, coords in ball_markup.items():
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
                if response.status_code == 200:
                    self.newprint(
                        f"Updated frame {frameid}, column {column} successfully.",
                        skipconsole=True,
                        event="updatecolumn1",
                    )
                else:
                    self.newprint(
                        f"Failed to update frame {frameid}, column {column}: {response.json()}",
                        skipconsole=True,
                        event="updatecolumn1",
                        level="error",
                    )

        return True


if __name__ == "__main__":
    c1 = Ball2DPositionConsumer(
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
        id="ball-2d-position-detection",
    )
    c1.threadstart()
