import requests
from ConsumerClass import Consumer
import random
from constants import Constants
from ultralytics import YOLO
import cv2
import torch
import json 
import pprint

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
            "ballx",
            "bally",
        ]
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
                    "videoid": videoid
                }
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
                raise Exception(f"Failed to get ball coordinates for frame {frameid}: {response.json()}")

        return (True, returnmap) if not missingframes else (False, self.groupframesintoranges(missingframes))

    def getzpointfromxandy(self, annotatedimagepath, xcoordinate, ycoordinate):
        # Dummy implementation for Z coordinate retrieval
        # In a real scenario, this would involve image processing to determine the Z coordinate
        return random.uniform(0, 100)  # Return a random Z value for demonstration

    def logicfunction(self, messagebody):
        videopath = requests.get(f"{self.server}/get-video-path-against-id", params={"videoId": messagebody["videoid"]}).json().get("videoPath", Constants.DEFAULT_VIDEO_PATH)
        startframeid = messagebody.get("startframeid", 0)
        endframeid = messagebody.get("endframeid", 1000)
        
        endframeimagepath = "/path/to/end/frame/image.jpg"  # Placeholder path
        
        ball_coordinates_status, ball_coordinates_data = self.getballcoordinates(startframeid, endframeid, messagebody["videoid"])

        if not ball_coordinates_status:
            print(f"Missing ball coordinates for frames: {ball_coordinates_data}")
            # TODO: Check if framestart and end being the same causes any issues downstream
            for (missingframestart, missingframeend) in ball_coordinates_data:
                self.placerequest(self.ballcoordinatescolumns, messagebody["requestid"], missingframestart, missingframeend)

            return False

        operation_status, annotated_image_path = self.annotate_coordinates_on_image_and_save(endframeimagepath, ball_coordinates_data)

        finalresult = dict()
        for frameid, coords in ball_coordinates_data.items():
            xcoordinate = coords.get("ballx", None)
            ycoordinate = coords.get("bally", None)
            zcoordinate = self.getzpointfromxandy(annotated_image_path, xcoordinate, ycoordinate) if xcoordinate is not None and ycoordinate is not None else None
            finalresult[frameid] = {
                "ballz": zcoordinate
            }

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
                        "videoid": videoId
                    }
                )
                if response.status_code == 200:
                    print(f"Updated frame {frameid}, column {column} successfully.")
                else:
                    print(f"Failed to update frame {frameid}, column {column}: {response.json()}")
                
        return True
    
if __name__ == "__main__":
    c1 = SpacePointSearch(rabbitmqusername=Constants.RABBITMQ_USERNAME, rabbitmqpassword=Constants.RABBITMQ_PASSWORD, id="space-point-searchs")
    c1.threadstart()