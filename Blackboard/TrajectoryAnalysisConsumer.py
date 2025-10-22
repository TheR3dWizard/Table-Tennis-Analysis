import requests
from ConsumerClass import Consumer
import random
from constants import Constants
from ultralytics import YOLO
import cv2
import torch
import json 
import pprint

class TrajectoryAnalysisConsumer(Consumer):
    def __init__(
        self,
        name="Trajectory Analysis",
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
            "ballxvector",
            "ballyvector",
            "ballzvector",
            "ballbounce",
        ]
        self.tablecoordinatescolumns = [
            "tablex1",
            "tabley1",
            "tablex2",
            "tabley2",
            "tablex3",
            "tabley3",
            "tablex4",
            "tabley4",
        ]
        self.ballcoordinatescolumns = [
            "ballx",
            "bally",
        ]
        self.joinserver()
    
    def groupframesintoranges(self, lst):
        # convert a list of integer frame ids into ranges whereever possible
        # example: [1,2,3,5,6,8] -> [(1,3),(5,6),(8)]
        if not lst:
            return []
        lst = sorted(set(lst))
        ranges = []
        start = prev = lst[0]
        for num in lst[1:]:
            if num == prev + 1:
                prev = num
            else:
                if start == prev:
                    ranges.append((start, start))
                else:
                    ranges.append((start, prev))
                start = prev = num
        if start == prev:
            ranges.append((start, start))
        else:
            ranges.append((start, prev))
        return ranges

    def gettablecoordinates(self, startframeid, endframeid, videoid):
        returnmap = dict()
        missingframes = []
        for frameid in range(startframeid, endframeid + 1):
            response = requests.post(
                f"{self.server}/checkandreturn",
                json={
                    "frameid": frameid,
                    "columns": self.tablecoordinatescolumns,
                    "videoid": videoid
                }
            )
            data = response.json()
            if response.status_code == 404 or not data:
                missingframes.append(frameid)
                continue

            if response.status_code == 200:
                for column in self.tablecoordinatescolumns:
                    if column not in data:
                        data[column] = None
                returnmap[frameid] = data
            else:
                raise Exception(f"Failed to get table coordinates for frame {frameid}: {response.json()}")

        return (True, returnmap) if not missingframes else (False, self.groupframesintoranges(missingframes))
    
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

    def logicfunction(self, messagebody):
        startframeid = messagebody.get("startframeid", 0)
        endframeid = messagebody.get("endframeid", 1000)

        ball_coordinates_status, ball_coordinates_data = self.getballcoordinates(startframeid, endframeid, messagebody["videoid"])
        table_coordinates_status, table_coordinates_data = self.gettablecoordinates(startframeid, endframeid, messagebody["videoid"])

        if not ball_coordinates_status:
            print(f"Missing ball coordinates for frames: {ball_coordinates_data}")
            # TODO: Check if framestart and end being the same causes any issues downstream
            for (missingframestart, missingframeend) in ball_coordinates_data:
                self.placerequest(self.ballcoordinatescolumns, messagebody["requestid"], missingframestart, missingframeend)

            return False
        
        if not table_coordinates_status:
            print(f"Missing table coordinates for frames: {table_coordinates_data}")
            # TODO: Check if framestart and end being the same causes any issues downstream
            for (missingframestart, missingframeend) in table_coordinates_data:
                self.placerequest(self.tablecoordinatescolumns, messagebody["requestid"], missingframestart, missingframeend)

            return False
        
        interpolated_ball_positions = dict() # Replace with actual function call
        '''
        interpolated_ball_positions format:
        {
            "frameid1": {"ballx": x1, "bally": y1},
            "frameid2": {"ballx": x2, "bally": y2},
            ...
        }
        '''

        ball_velocities = dict() # Replace with actual function call
        '''
        ball_velocities format:
        {
            "frameid1": {"ballxvector": vx1, "ballyvector": vy1},
            "frameid2": {"ballxvector": vx2, "ballyvector": vy2},
            ...
        }
        '''

        bounceframes = list() # Replace with actual function call
        '''
        bounceframes format:
        [frameid1, frameid2, ...]
        '''


        self.saveresult(interpolated_ball_positions, ball_velocities, bounceframes, messagebody["videoid"])
        
        return True

    def saveresult(self, interpolated_ball_positions, ball_velocities, bounceframes, videoId):
        # TODO: Combine all functions below into one function to reduce repetitive code
        self.saveballpositionresult(interpolated_ball_positions, videoId)
        self.saveballvelocityresult(ball_velocities, videoId)
        self.saveballbounce(bounceframes, videoId)

    def saveballvelocityresult(self, ball_velocities, videoId):
        print("Executing saveballvelocityresult.... for ", videoId)

        for frameid, vectors in ball_velocities.items():
            for column, value in vectors.items():
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

    def saveballpositionresult(self, interpolated_ball_positions, videoId):
        print("Executing saveballpositionresult.... for ", videoId)

        for frameid, coords in interpolated_ball_positions.items():
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
                
    def saveballbounce(self, bounceframes, videoId):
        print("Executing saveballbounce.... for ", videoId)

        for frameid in bounceframes:
            response = requests.post(
                f"{self.server}/updatecolumn",
                json={
                    "frameid": int(frameid),
                    "column": "ballbounce",
                    "value": True,
                    "videoid": videoId
                }
            )
            if response.status_code == 200:
                print(f"Updated frame {frameid}, column ballbounce successfully.")
            else:
                print(f"Failed to update frame {frameid}, column ballbounce: {response.json()}")

if __name__ == "__main__":
    c1 = TrajectoryAnalysisConsumer(rabbitmqusername=Constants.RABBITMQ_USERNAME, rabbitmqpassword=Constants.RABBITMQ_PASSWORD, id="trajectory-analysis")
    c1.threadstart()