from ConsumerClass import Consumer
import requests
from tt_pose_heatmap import analyze_video
import pprint
from constants import Constants


class PlayerHeatmapConsumer(Consumer):
    def __init__(
        self,
        name="Player Heatmap Generation",
        id=None,
        queuename=None,
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
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
        self.HEATMAP_DEFAULT_VIDEO = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/assets/rallies_02.mp4"
        self.HEATMAP_TRACKER = "bytetrack.yaml"
        self.HEATMAP_OUTPUT_PATH = "/Users/akashshanmugaraj/Documents/Personal Projects/Table-Tennis-Analysis/outputs"
        self.HEATMAP_TRACKPOINT_THRESHOLD = 50
        self.processable_columns = [
            "player1x",
            "player1y",
            "player1z",
            "player2x",
            "player2y",
            "player2z",
        ]
        self.joinserver()

    def saveresult(self, videoId, resultMap):
        self.newprint("Executing saveresult....")
        response = requests.post(
            f"{self.server}/update-player-coordinates",
            json={"videoId": videoId, "both_player_coords_map": resultMap},
        )
        if response.status_code == 200:
            self.newprint("Player coordinates updated successfully.")
        else:
            self.newprint(f"Failed to update player coordinates: {response.json()}")
        return response.json()

    def logicfunction(self, messagebody):
        self.newprint(f"Processing message: {messagebody}")
        videopath = (
            requests.get(
                f"{self.server}/get-video-path-against-id",
                params={"videoId": messagebody["videoid"]},
            )
            .json()
            .get("videoPath", Constants.DEFAULT_VIDEO_PATH)
        )
        self.newprint(f"Video path retrieved: {videopath}")
        framedatamap, heatmapimagepath = analyze_video(
            video=videopath if videopath else Constants.HEATMAP_DEFAULT_VIDEO,
            start_frame=messagebody.get("startframeid", 0),
            end_frame=messagebody.get("endframeid", 1000),
            model=Constants.YOLO11N_POSE_WEIGHTS_PATH,
            tracker=self.HEATMAP_TRACKER,
            confidence=0.3,
            device="cpu",
            point_radius=6,
            sigma=8.0,
            out_dir=Constants.DEFAULT_OUTPUT_FOLDER_PATH,
            num_players=2,
            min_track_points=self.HEATMAP_TRACKPOINT_THRESHOLD,
        )

        # Save results to the database
        self.saveresult(messagebody["videoid"], framedatamap)
        pprint.pprint(framedatamap)
        self.newprint(f"Heatmap image saved at: {heatmapimagepath}")

        return True


if __name__ == "__main__":
    c1 = PlayerHeatmapConsumer(
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
        id="player-heatmap-consumer",
    )
    c1.threadstart()
