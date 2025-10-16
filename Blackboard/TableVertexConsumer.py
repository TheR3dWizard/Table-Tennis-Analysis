import requests
from ConsumerClass import Consumer
import random
import cv2
import numpy as np
from ultralytics import YOLO
import pprint

class TableVertexConsumer(Consumer):
    def __init__(
        self,
        name="Table Vertex Detection",
        id=None,
        queuename=None,
        rabbitmqusername="default",
        rabbitmqpassword="default",
        serverurl="http://localhost:6060",
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
        self.TABLE_DETECTION_WEIGHTS_PATH = "../weights/TableDetection.pt"
        self.TABLE_DETECTION_DEFAULT_VIDEO = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/assets/rallies_02.mp4"
        self.processable_columns = [
            "tablex1",
            "tabley1",
            "tablex2",
            "tabley2",
            "tablex3",
            "tabley3",
            "tablex4",
            "tabley4",
        ]
        self.joinserver()
    
    def computerandomtablevertices(self):
        print("Executing computerandomtablevertices....")
        x1, y1 = random.uniform(0,1000), random.uniform(0,1000)
        x2, y2 = random.uniform(0,1000), random.uniform(0,1000)
        x3, y3 = random.uniform(0,1000), random.uniform(0,1000)
        x4, y4 = random.uniform(0,1000), random.uniform(0,1000)
        '''
        tablex1 FLOAT,
        tabley1 FLOAT,
        tablex2 FLOAT,
        tabley2 FLOAT,
        tablex3 FLOAT,
        tabley3 FLOAT,
        tablex4 FLOAT,
        tabley4 FLOAT,'''
        return {
            "tablex1": x1,
            "tabley1": y1,
            "tablex2": x2,
            "tabley2": y2,
            "tablex3": x3,
            "tabley3": y3,
            "tablex4": x4,
            "tabley4": y4,
        }

    def yolo_on_video(self, model, video, start_frame, end_frame):
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)


        frame_num = start_frame
        all_results = []
        
        # Detect device: use MPS for Mac GPU, otherwise CPU
        import torch
        if torch.backends.mps.is_available():
            device = "cpu"
        else:
            device = "cpu"
        
        print(f"Using device: {device}")
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_num > end_frame:
                break

            # Run YOLO inference with appropriate device
            results = model(frame, stream=True, device=device)

            for r in results:
                all_results.append(r)
                annotated_frame = r.plot()
                # cv2.imshow("YOLO Pose - Full", annotated_frame)

                keypoints = r.keypoints.cpu().numpy()  # (num_instances, num_keypoints, 3)
                if len(keypoints) > 0:
                    table_corners = keypoints[0][:, :2]  # first instance, all keypoints, x,y only
                    print(f"Frame {frame_num}: Table corners (normalized): {table_corners}")

            frame_num += 1


        cap.release()
        cv2.destroyAllWindows()
        return all_results
    
    def average_results(results):
        if len(results) == 0:
            print("No results to average")
            return None

        sum_corners = np.zeros((4, 2), dtype=np.float32)
        count = 0

        for r in results:
            keypoints = r.keypoints.cpu().numpy()
            table_corners = None

            # Case 1: use keypoints if available
            if len(keypoints) > 0:
                candidate = keypoints[0][:, :2]   # take first detection (x, y only)
                if candidate.shape == (4, 2):
                    table_corners = candidate

            # Case 2: fallback to bounding box
            if table_corners is None:
                if hasattr(r, "boxes") and len(r.boxes) > 0:
                    box = r.boxes[0].xyxy.cpu().numpy()[0]  # (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box
                    table_corners = np.array([
                        [x1, y1],  # top-left
                        [x2, y1],  # top-right
                        [x2, y2],  # bottom-right
                        [x1, y2],  # bottom-left
                    ], dtype=np.float32)

            # Add if we have valid corners
            if table_corners is not None and table_corners.shape == (4, 2):
                sum_corners += table_corners
                count += 1

        if count == 0:
            return None

        avg_corners = sum_corners / count
        return avg_corners

    def construct_return_object(self, returnobject, frameid, tablecoordinates):
        framereturnobject = {
            "tablex1": tablecoordinates[0],
            "tabley1": tablecoordinates[1],
            "tablex2": tablecoordinates[2],
            "tabley2": tablecoordinates[3],
            "tablex3": tablecoordinates[4],
            "tabley3": tablecoordinates[5],
            "tablex4": tablecoordinates[6],
            "tabley4": tablecoordinates[7],
        }

        returnobject[frameid] = framereturnobject

        return returnobject
    
    def saveresult(self, videoId, resultMap):
        print("Executing saveresult.... for ", videoId)
        pprint.pprint(resultMap)
        for frameid, result in resultMap.items():
            for column, value in result.items():
                response = requests.post(
                    f"{self.server}/updatecolumn",
                    json={
                        "frameid": int(frameid),
                        "column": column,
                        "value": float(value),  # Convert numpy float32 to Python float
                        "videoid": videoId
                    }
                )
                if response.status_code == 200:
                    print(f"Updated frame {frameid}, column {column} successfully.")
                else:
                    print(f"Failed to update frame {frameid}, column {column}: {response.json()}")
    
    def logicfunction(self, messagebody):
        videopath = requests.get(f"{self.server}/get-video-path-against-id", params={"videoId": messagebody["targetid"]}).json().get("videoPath", self.TABLE_DETECTION_DEFAULT_VIDEO)
        model = YOLO(self.TABLE_DETECTION_WEIGHTS_PATH)
        startframeid = messagebody.get("startframeid", 0)
        endframeid = messagebody.get("endframeid", 1000)
        results = self.yolo_on_video(model, videopath, startframeid, endframeid)
        if not results:
            print("No results from YOLO inference")
            return self.computerandomtablevertices()

        # avg_corners = self.average_results(results)
        # if avg_corners is not None:
        #     print(f"Average table corners (normalized): {avg_corners}")

        returnobject = dict()
        for i in results:
            if not len(i.boxes):
                self.construct_return_object(returnobject, startframeid, (0,0,0,0,0,0,0,0))
            for box in i.boxes:
                box_coords = box.xyxy.cpu().numpy()[0]  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = box_coords
                corners = (x1, y1, x2, y1, x2, y2, x1, y2)  # top-left, top-right, bottom-right, bottom-left
                self.construct_return_object(returnobject, startframeid, corners)
            startframeid += 1

        print("Saving results to database...")
        self.saveresult(messagebody["videoid"], returnobject)            


if __name__ == "__main__":
    c1 = TableVertexConsumer(rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword", id="table-vertex-detection-consumer")
    c1.threadstart()