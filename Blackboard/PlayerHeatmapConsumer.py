from ConsumerClass import Consumer
import requests
from tt_pose_heatmap import analyze_video
import pprint
from constants import Constants
from tqdm import tqdm


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

    def saveresult(self, videoId, resultMap, startframeid, endframeid):
        self.newprint("Executing saveresult....")

        # Calculate total frames to update
        total_frames = len(resultMap)
        frames_updated = 0
        
        # Create progress bar with custom format
        progress_bar = tqdm(
            total=total_frames,
            desc="Saving player coordinates",
            unit="frame",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| ({n_fmt}/{total_fmt}) [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for frameid, coords in resultMap.items():
            frame_success = True
            for column, value in coords.items():
                if column == "combinedheatmappath":
                    # remove trailing \n 
                    value = value.replace("\n", "")
                    # value is path of an image, make it a database friendly pathstring
                else:
                    value = float(value)
                    
                response = requests.post(
                    f"{self.server}/updatecolumn",
                    json={
                        "frameid": int(frameid),
                        "column": column,
                        "value": (
                            value
                        ),  # Convert numpy float32 to Python float
                        "videoid": videoId,
                    },
                )
                if response.status_code != 200:
                    frame_success = False
                    self.newprint(
                        f"Failed to update frame {frameid}, column {column}: {response.json()}",
                        event="updateplayercoords",
                        level="error",
                    )
            
            # Update progress bar after processing all columns for this frame
            if frame_success:
                frames_updated += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"Updated": f"{frames_updated}/{total_frames}"})

        progress_bar.close()
        self.newprint(f"Successfully updated {frames_updated}/{total_frames} frames", event="saveresult_complete", level="info")
        
        return {"status": "success", "frames_updated": frames_updated, "total_frames": total_frames}

    def logicfunction(self, messagebody):
        self.newprint(f"Processing message: {messagebody}", skipconsole=True)
        videopath = (
            requests.get(
                f"{self.server}/get-video-path-against-id",
                params={"videoId": messagebody["videoid"]},
            )
            .json()
            .get("videoPath", Constants.DEFAULT_VIDEO_PATH)
        )
        self.newprint(f"Video path retrieved: {videopath}", skipconsole=True)
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
        self.saveresult(messagebody["videoid"], framedatamap, startframeid=messagebody.get("startframeid", 0), endframeid=messagebody.get("endframeid", 1000))
        self.newprint(f"Heatmap image saved at: {heatmapimagepath}")

        return True


if __name__ == "__main__":
    c1 = PlayerHeatmapConsumer(
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
        id="player-heatmap-consumer",
    )
    c1.threadstart()
