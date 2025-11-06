import requests
from ConsumerClass import Consumer
from constants import Constants
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from KalmanTRackerClass import KalmanTracker
import numpy as np
import pandas as pd

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
                    "videoid": videoid,
                },
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
                raise Exception(
                    f"Failed to get table coordinates for frame {frameid}: {response.json()}"
                )

        return (
            (True, returnmap)
            if not missingframes
            else (False, self.groupframesintoranges(missingframes))
        )

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

    def interpolate_missing_frames(
        self,
        segment_frames,
        segment_positions,
        valid_frames,
        valid_positions,
        max_gap=5,
    ):
        """Interpolate missing frames using cubic spline if enough points available."""

        if len(valid_positions) < 4:
            print("Warning: Less than 4 valid points, skipping interpolation")
            return segment_positions

        # Convert valid_positions to numpy array to ensure consistent indexing
        valid_positions_array = np.array(valid_positions)

        interp_x = interp1d(
            valid_frames,
            valid_positions_array[:, 0],
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        interp_y = interp1d(
            valid_frames,
            valid_positions_array[:, 1],
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        interpolated_positions = segment_positions.copy()
        interpolated_count = 0

        for i, frame_num in enumerate(segment_frames):
            if segment_positions[i] is None:
                prev_valid = None
                next_valid = None

                for j in range(i - 1, -1, -1):
                    if segment_positions[j] is not None:
                        prev_valid = segment_frames[j]
                        break

                for j in range(i + 1, len(segment_frames)):
                    if segment_positions[j] is not None:
                        next_valid = segment_frames[j]
                        break

                if prev_valid is not None and next_valid is not None:
                    if (next_valid - prev_valid) <= max_gap:
                        x_interp = float(interp_x(frame_num))
                        y_interp = float(interp_y(frame_num))
                        interpolated_positions[i] = [x_interp, y_interp]
                        interpolated_count += 1

        return interpolated_positions

    def smooth_trajectory_kalman(
        self, segment_frames, interpolated_positions, valid_positions
    ):
        """Apply Kalman filtering for trajectory smoothing."""

        tracker = KalmanTracker()
        # print("Kalman tracker initialized")

        first_valid_pos = next(pos for pos in interpolated_positions if pos is not None)
        tracker.state[:2] = first_valid_pos
        # print(f"Set initial position: x={first_valid_pos[0]:.2f}, y={first_valid_pos[1]:.2f}")

        if len(valid_positions) >= 2:
            valid_indices = [
                i for i, pos in enumerate(interpolated_positions) if pos is not None
            ]
            if len(valid_indices) >= 2:
                pos1 = interpolated_positions[valid_indices[0]]
                pos2 = interpolated_positions[valid_indices[1]]
                dt_frames = (
                    segment_frames[valid_indices[1]] - segment_frames[valid_indices[0]]
                )
                # Protect against division by zero
                if dt_frames > 0:
                    vx_init = (pos2[0] - pos1[0]) / dt_frames
                    vy_init = (pos2[1] - pos1[1]) / dt_frames
                    tracker.state[2:] = [vx_init, vy_init]
                    # print(f"Set initial velocity: vx={vx_init:.2f}, vy={vy_init:.2f}")

        for i, pos in enumerate(interpolated_positions):
            measurement = np.array(pos) if pos is not None else None
            tracker.predict()
            tracker.update(measurement)

        smoothed_positions = np.array(tracker.history)
        confidence_scores = tracker.confidence
        # print(f"Generated {len(smoothed_positions)} smoothed positions")

        if len(smoothed_positions) >= 7:
            window_length = min(7, len(smoothed_positions))
            if window_length % 2 == 0:
                window_length -= 1
            # print(f"Applying Savitzky-Golay filter with window length {window_length}")
            if window_length >= 3:
                smoothed_positions[:, 0] = savgol_filter(
                    smoothed_positions[:, 0], window_length=window_length, polyorder=2
                )
                smoothed_positions[:, 1] = savgol_filter(
                    smoothed_positions[:, 1], window_length=window_length, polyorder=2
                )
                # print("Savitzky-Golay filter applied to smooth trajectory")

        return smoothed_positions, confidence_scores

    # def correct_bounces_with_table(self, smoothed_positions, table_coords):
    #     """Corrects the trajectory by aligning suspected bounce points with the table's y-coordinates."""
    #     corrected_positions = smoothed_positions.copy()

    #     # extract y-coordinates correctly from table_coords dict
    #     top_y = min(
    #         table_coords["tabley1"],
    #         table_coords["tabley2"],
    #         table_coords["tabley3"],
    #         table_coords["tabley4"],
    #     )
    #     bottom_y = max(
    #         table_coords["tabley1"],
    #         table_coords["tabley2"],
    #         table_coords["tabley3"],
    #         table_coords["tabley4"],
    #     )
    #     # print(f"Table coordinates: top_y={top_y}, bottom_y={bottom_y}")

    #     if len(corrected_positions) < 3:
    #         # print("Too few positions to correct bounces, returning unchanged")
    #         return corrected_positions

    #     vy = np.gradient(corrected_positions[:, 1])
    #     bounce_count = 0

    #     for i in range(1, len(corrected_positions) - 1):
    #         if vy[i - 1] * vy[i + 1] < 0:
    #             if (
    #                 abs(corrected_positions[i, 1] - top_y) < 15
    #                 or abs(corrected_positions[i, 1] - bottom_y) < 15
    #             ):
    #                 if abs(corrected_positions[i, 1] - top_y) < abs(
    #                     corrected_positions[i, 1] - bottom_y
    #                 ):
    #                     corrected_positions[i, 1] = top_y
    #                     # print(f"Corrected bounce at frame {i} to table top (y={top_y})")
    #                 else:
    #                     corrected_positions[i, 1] = bottom_y
    #                     # print(f"Corrected bounce at frame {i} to table bottom (y={bottom_y})")
    #                 bounce_count += 1

    #     # print(f"Corrected {bounce_count} bounce points")
    #     return corrected_positions

    # def detect_bounce_points(
    #     self,
    #     smoothed_positions,
    #     table_coords,
    #     startframeid,
    #     proximity_threshold=15,
    #     min_velocity_change=0.5,
    #     segment_frames=None,
    # ):
    #     """
    #     Detect bounce points and return only frame IDs in bounceframes format: [frameid1, frameid2, ...]
    #     """
    #     if len(smoothed_positions) < 5:
    #         return []
    #     y = smoothed_positions[:, 1]
    #     vy = np.gradient(y)

    #     # extract y-coordinates correctly from table_coords dict
    #     top_y = min(
    #         table_coords["tabley1"],
    #         table_coords["tabley2"],
    #         table_coords["tabley3"],
    #         table_coords["tabley4"],
    #     )
    #     bottom_y = max(
    #         table_coords["tabley1"],
    #         table_coords["tabley2"],
    #         table_coords["tabley3"],
    #         table_coords["tabley4"],
    #     )

    #     bounce_frames = []
    #     last_bounce_frame = -999
    #     # Default: segment_frames = [start_frame, start_frame+1, ...]
    #     if segment_frames is None or len(segment_frames) != len(y):
    #         raise ValueError(
    #             "segment_frames must be provided and match the length of smoothed_positions."
    #         )

    #     for i in range(2, len(y) - 2):
    #         proximity_top = abs(y[i] - top_y) < proximity_threshold
    #         proximity_bottom = abs(y[i] - bottom_y) < proximity_threshold
    #         if (
    #             (y[i] < y[i - 1])
    #             and (y[i] < y[i + 1])
    #             and (proximity_top or proximity_bottom)
    #         ):
    #             v_change = abs(vy[i - 1] - vy[i + 1])
    #             if v_change >= min_velocity_change and i - last_bounce_frame > 3:
    #                 bounce_frames.append(segment_frames[i])
    #                 last_bounce_frame = i
    #     shifted_bounce_frames = [bf for bf in bounce_frames]
    #     return shifted_bounce_frames




    def correct_bounces_with_table(self, smoothed_positions, table_coords):
        """
        Corrects the trajectory by aligning suspected bounce points 
        with the table's top or bottom y-coordinates.
        """
        if smoothed_positions is None or len(smoothed_positions) < 3:
            return smoothed_positions

        corrected_positions = smoothed_positions.copy()

        # Extract y-coordinates safely from the table_coords dictionary
        table_y_values = [
            table_coords.get("tabley1"),
            table_coords.get("tabley2"),
            table_coords.get("tabley3"),
            table_coords.get("tabley4"),
        ]
        if None in table_y_values:
            raise ValueError("table_coords must contain keys 'tabley1' to 'tabley4'.")

        top_y = min(table_y_values)
        bottom_y = max(table_y_values)

        vy = np.gradient(corrected_positions[:, 1])
        bounce_count = 0

        for i in range(1, len(corrected_positions) - 1):
            # Detect sign change in vertical velocity (potential bounce)
            if vy[i - 1] * vy[i + 1] < 0:
                y_pos = corrected_positions[i, 1]
                dist_top = abs(y_pos - top_y)
                dist_bottom = abs(y_pos - bottom_y)

                if dist_top < 15 or dist_bottom < 15:
                    corrected_positions[i, 1] = top_y if dist_top < dist_bottom else bottom_y
                    bounce_count += 1

        # Optionally log or return bounce count if needed
        # print(f"Corrected {bounce_count} bounce points.")
        return corrected_positions

    
    def detect_bounce_points(self, smoothed_positions, table_coords,
                            proximity_threshold=250,  # Increased significantly
                            min_velocity_change=8.0,  # Increased to filter noise
                            min_frame_gap=10,  # Increased to avoid duplicate detections
                            segment_frames=None):
        """
        Detect bounce frames, handling missing ('-1') values robustly.
        Accepts table_coords either as:
          - a dict with keys 'tabley1'..'tabley4' (as returned by gettablecoordinates[frameid])
          - or as an iterable of (x,y) pairs.
        """
        if segment_frames is None or len(segment_frames) != len(smoothed_positions):
            raise ValueError('segment_frames must be provided and have the same length as smoothed_positions')
        
        # Ensure numpy array input
        smoothed_positions = np.asarray(smoothed_positions)
        
        # Normalize table_coords to a list of numeric y-values
        if isinstance(table_coords, dict):
            try:
                table_y_values = [
                    float(table_coords["tabley1"]),
                    float(table_coords["tabley2"]),
                    float(table_coords["tabley3"]),
                    float(table_coords["tabley4"]),
                ]
            except Exception:
                raise ValueError("table_coords dict must contain numeric 'tabley1'..'tabley4' values")
        else:
            # assume iterable of (x,y) pairs
            try:
                table_y_values = [float(coord[1]) for coord in table_coords]
            except Exception:
                raise ValueError("table_coords must be a dict or an iterable of (x,y) pairs")
        
        top_y = min(table_y_values)
        bottom_y = max(table_y_values)
        
        # Sort by frame number chronologically
        sorted_indices = np.argsort(segment_frames)
        smoothed_positions = smoothed_positions[sorted_indices]
        segment_frames = [segment_frames[i] for i in sorted_indices]
        
        # Convert y to float dtype
        y = smoothed_positions[:, 1].astype(float)
        
        # Replace -1s with np.nan for interpolation
        y[y == -1] = np.nan
        
        # Interpolate missing values
        s = pd.Series(y)
        y_interp = s.interpolate(limit_direction="both").values
        
        vy = np.gradient(y_interp)
        
        bounce_frames = []
        last_bounce_frame = -999
        
        for i in range(2, len(y_interp) - 2):
            y_curr = y_interp[i]
            proximity_top = abs(y_curr - top_y) < proximity_threshold
            proximity_bottom = abs(y_curr - bottom_y) < proximity_threshold
            
            # Look for local maxima (peaks in y, where ball reverses downward motion)
            is_local_maximum = (y_curr > y_interp[i-1]) and (y_curr > y_interp[i+1])
            
            if is_local_maximum and (proximity_top or proximity_bottom):
                v_change = abs(vy[i-1] - vy[i+1])
                sign_change = vy[i-1] * vy[i+1] < 0
                
                # Require significant velocity change AND sign change for bounces
                if sign_change and v_change >= min_velocity_change and (i - last_bounce_frame) > min_frame_gap:
                    bounce_frames.append(segment_frames[i])
                    last_bounce_frame = i
        
        print("Detected bounce frames:", bounce_frames)
        return bounce_frames

    def logicfunction(self, messagebody):
        startframeid = messagebody.get("startframeid", 0)
        endframeid = messagebody.get("endframeid", 1000)

        ball_coordinates_status, ball_coordinates_data = self.getballcoordinates(
            startframeid, endframeid, messagebody["videoid"]
        )
        table_coordinates_status, table_coordinates_data = self.gettablecoordinates(
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

        if not table_coordinates_status:
            print(f"Missing table coordinates for frames: {table_coordinates_data}")
            # TODO: Check if framestart and end being the same causes any issues downstream
            for missingframestart, missingframeend in table_coordinates_data:
                self.placerequest(
                    self.tablecoordinatescolumns,
                    messagebody["requestid"],
                    missingframestart,
                    missingframeend,
                    videoid=messagebody["videoid"],
                )

            return False

        segment_frames = list(range(startframeid, endframeid + 1))
        segment_positions = []
        valid_frames = []
        valid_positions = []
        for frame_id in segment_frames:
            coords = ball_coordinates_data.get(frame_id)
            if coords and coords["ballx"] is not None and coords["bally"] is not None:
                seg = [coords["ballx"], coords["bally"]]
                segment_positions.append(seg)
                valid_frames.append(frame_id)
                valid_positions.append(seg)
            else:
                segment_positions.append(None)

        interpolated_ball_positions = self.interpolate_missing_frames(
            segment_frames, segment_positions, valid_frames, valid_positions, max_gap=5
        )
        # Convert interpolated_ball_positions (list of lists) to dict mapping frameid to {"ballx": ..., "bally": ...}
        interpolated_ball_positions_dict = {}
        for i, frame_id in enumerate(segment_frames):
            pos = interpolated_ball_positions[i]
            if pos is not None:
                interpolated_ball_positions_dict[frame_id] = {
                    "ballx": pos[0],
                    "bally": pos[1],
                }
            else:
                interpolated_ball_positions_dict[frame_id] = {
                    "ballx": None,
                    "bally": None,
                }

        smoothed_positions, confidence_scores = self.smooth_trajectory_kalman(
            segment_frames, interpolated_ball_positions, valid_positions
        )

        ball_velocities = {}
        if len(smoothed_positions) >= 2:
            for i in range(1, len(smoothed_positions)):
                vx = smoothed_positions[i][0] - smoothed_positions[i - 1][0]
                vy = smoothed_positions[i][1] - smoothed_positions[i - 1][1]
                frameid = segment_frames[i]
                ball_velocities[frameid] = {"ballxvector": vx, "ballyvector": vy}
        else:
            for frame_id in segment_frames:
                ball_velocities[frame_id] = {"ballxvector": 0.0, "ballyvector": 0.0}

        valid_interpolated_positions = []
        valid_interpolated_frames = []
        for i, pos in enumerate(interpolated_ball_positions):
            if pos is not None:
                valid_interpolated_positions.append(pos)
                valid_interpolated_frames.append(segment_frames[i])

        # Convert to numpy array only if we have valid positions
        if len(valid_interpolated_positions) > 0:
            bounceframes = self.detect_bounce_points(
                np.array(valid_interpolated_positions),
                table_coordinates_data[startframeid],
                segment_frames=valid_interpolated_frames
            )
        else:
            bounceframes = []
        """
        bounceframes format:
        [frameid1, frameid2, ...]
        """

        self.saveresult(
            interpolated_ball_positions_dict,
            ball_velocities,
            bounceframes,
            messagebody["videoid"],
        )

        return True

    def saveresult(
        self, interpolated_ball_positions, ball_velocities, bounceframes, videoId
    ):
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
                        "videoid": videoId,
                    },
                )
                if response.status_code == 200:
                    print(f"Updated frame {frameid}, column {column} successfully.")
                else:
                    print(
                        f"Failed to update frame {frameid}, column {column}: {response.json()}"
                    )

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
                        "videoid": videoId,
                    },
                )
                if response.status_code == 200:
                    print(f"Updated frame {frameid}, column {column} successfully.")
                else:
                    print(
                        f"Failed to update frame {frameid}, column {column}: {response.json()}"
                    )

    def saveballbounce(self, bounceframes, videoId):
        print("Executing saveballbounce.... for ", videoId)
        for frameid in bounceframes:
            print("Updating bounce for frameid: ", frameid)
            response = requests.post(
                f"{self.server}/updatecolumn",
                json={
                    "frameid": int(frameid),
                    "column": "ballbounce",
                    "value": True,
                    "videoid": videoId,
                },
            )
            if response.status_code == 200:
                print(f"Updated frame {frameid}, column ballbounce successfully.")
            else:
                print(
                    f"Failed to update frame {frameid}, column ballbounce: {response.json()}"
                )
        print("Finished updating ballbounce for all frames.")


if __name__ == "__main__":
    c1 = TrajectoryAnalysisConsumer(
        rabbitmqusername=Constants.RABBITMQ_USERNAME,
        rabbitmqpassword=Constants.RABBITMQ_PASSWORD,
        id="trajectory-analysis",
    )
    c1.threadstart()