from ConsumerClass import Consumer
import requests
from tt_pose_heatmap import analyze_video
import pprint

class PlayerHeatmapConsumer(Consumer):
    def __init__(
        self,
        name="Player Heatmap Generation",
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
        self.HEATMAP_DEFAULT_VIDEO = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/assets/rallies_02.mp4"
        self.HEATMAP_MODEL_PATH = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/yolo11n-pose.pt"
        self.HEATMAP_TRACKER = "bytetrack.yaml"
        self.HEATMAP_OUTPUT_PATH = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/outputs"
        self.HEATMAP_TRACKPOINT_THRESHOLD = 50
    
    def saveresult(self, videoId, resultMap):
        print("Executing saveresult....")
        response = requests.post(
            f"{self.server}/update-player-coordinates",
            json={
                "videoId": videoId,
                "both_player_coords_map": resultMap
            }
        )
        if response.status_code == 200:
            print("Player coordinates updated successfully.")
        else:
            print(f"Failed to update player coordinates: {response.json()}")
        return response.json()

    def logicfunction(self, messagebody):
        print(f"Processing message: {messagebody}")
        framedatamap, heatmapimagepath = analyze_video(
            video=self.HEATMAP_DEFAULT_VIDEO,
            start_frame=messagebody.get("startframeid", 0),
            end_frame=messagebody.get("endframeid", 1000),
            model=self.HEATMAP_MODEL_PATH,
            tracker=self.HEATMAP_TRACKER,
            confidence=0.3,
            device="cpu",
            point_radius=6,
            sigma=8.0,
            out_dir=self.HEATMAP_OUTPUT_PATH,
            num_players=2,
            min_track_points=self.HEATMAP_TRACKPOINT_THRESHOLD
        )
        
        # Save results to the database
        # TODO: Replace 1 with actual videoId from messagebody 
        self.saveresult(1, framedatamap)
        pprint.pprint(framedatamap)
        print(f"Heatmap image saved at: {heatmapimagepath}")

        # TODO: Implement success notification
        # Notify success
        # self.placesuccess(
        #     requestid=messagebody["requestid"],
        #     requesterid=messagebody["requesterid"],
        #     targetid=self.id,
        #     requestorqueue=messagebody["returnqueue"],
        #     returnmessageid=messagebody["returnmessageid"],
        #     startframeid=messagebody.get("startframeid", None),
        #     endframeid=messagebody.get("endframeid", None)
        # )
        return True

if __name__ == "__main__":
    c1 = PlayerHeatmapConsumer(rabbitmqusername="pw1tt", rabbitmqpassword="securerabbitmqpassword", id="player-heatmap-consumer")
    c1.threadstart()