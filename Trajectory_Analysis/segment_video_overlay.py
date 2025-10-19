import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os

class SegmentKalmanTracker:
    """Simplified Kalman filter for segment-based ball tracking."""
    def __init__(self, dt=1.0, process_noise=0.5, measurement_noise=3.0):
        self.dt = dt
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 0.95, 0], [0, 0, 0, 0.95]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 50
        self.history = []
        self.confidence = []
        print("Initialized Kalman tracker with dt=%.2f, process_noise=%.2f, measurement_noise=%.2f" % (dt, process_noise, measurement_noise))

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        print("Predicted next state: x=%.2f, y=%.2f, vx=%.2f, vy=%.2f" % tuple(self.state))
        return self.state[:2].copy()

    def update(self, measurement):
        if measurement is None:
            self.history.append(self.state[:2].copy())
            self.confidence.append(0.3)
            print("No measurement provided, using predicted state")
            return
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.history.append(self.state[:2].copy())
        self.confidence.append(1.0)
        print("Updated state with measurement: x=%.2f, y=%.2f" % tuple(measurement))

def load_ball_markup_data(json_filename):
    """Load ball markup data from JSON file."""
    print(f"Loading ball markup data from {json_filename}...")
    
    try:
        with open(json_filename, "r") as f:
            data = json.load(f)
        print(f"Successfully loaded JSON data with {len(data)} frames")
        return data
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        raise

def extract_segment_data(data, start_frame, end_frame, segment_label):
    """Extract trajectory data for a specific segment."""
    print(f"Extracting data for segment {segment_label}: frames {start_frame} to {end_frame}")
    
    segment_frames = []
    segment_positions = []
    valid_frames = []
    
    for frame_num in range(start_frame, end_frame + 1):
        frame_str = str(frame_num)
        segment_frames.append(frame_num)
        
        if frame_str in data:
            x = data[frame_str]["x"]
            y = data[frame_str]["y"]
            print(f"Frame {frame_num}: x={x}, y={y}")
            
            if x != -1 and y != -1:
                segment_positions.append([x, y])
                valid_frames.append(frame_num)
            else:
                segment_positions.append(None)
                print(f"Frame {frame_num}: Invalid position (x=-1, y=-1)")
        else:
            segment_positions.append(None)
            print(f"Frame {frame_num}: No data found")
    
    valid_positions = np.array([pos for pos in segment_positions if pos is not None])
    
    print(f"Found {len(valid_positions)} valid ball positions out of {len(segment_frames)} frames")
    print(f"Coverage: {len(valid_positions)/len(segment_frames)*100:.1f}%")
    
    return np.array(segment_frames), segment_positions, np.array(valid_frames), valid_positions

def interpolate_missing_frames(segment_frames, segment_positions, valid_frames, valid_positions, max_gap=5):
    """Interpolate missing frames using cubic spline if enough points available."""
    print(f"Starting interpolation for {len(segment_frames)} frames with max gap {max_gap}")
    
    if len(valid_positions) < 4:
        print("Warning: Less than 4 valid points, skipping interpolation")
        return segment_positions
    
    print(f"Interpolating missing frames (max gap: {max_gap} frames)...")
    
    interp_x = interp1d(valid_frames, valid_positions[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
    interp_y = interp1d(valid_frames, valid_positions[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    interpolated_positions = segment_positions.copy()
    interpolated_count = 0
    
    for i, frame_num in enumerate(segment_frames):
        if segment_positions[i] is None:
            prev_valid = None
            next_valid = None
            
            for j in range(i-1, -1, -1):
                if segment_positions[j] is not None:
                    prev_valid = segment_frames[j]
                    break
            
            for j in range(i+1, len(segment_frames)):
                if segment_positions[j] is not None:
                    next_valid = segment_frames[j]
                    break
            
            if prev_valid is not None and next_valid is not None:
                if (next_valid - prev_valid) <= max_gap:
                    x_interp = float(interp_x(frame_num))
                    y_interp = float(interp_y(frame_num))
                    interpolated_positions[i] = [x_interp, y_interp]
                    interpolated_count += 1
                    print(f"Interpolated frame {frame_num}: x={x_interp:.2f}, y={y_interp:.2f}")
    
    print(f"Interpolated {interpolated_count} missing frames")
    return interpolated_positions

def smooth_trajectory_kalman(segment_frames, interpolated_positions, valid_positions):
    """Apply Kalman filtering for trajectory smoothing."""
    print("Applying Kalman filter for trajectory smoothing...")
    
    tracker = SegmentKalmanTracker()
    print("Kalman tracker initialized")
    
    first_valid_pos = next(pos for pos in interpolated_positions if pos is not None)
    tracker.state[:2] = first_valid_pos
    print(f"Set initial position: x={first_valid_pos[0]:.2f}, y={first_valid_pos[1]:.2f}")
    
    if len(valid_positions) >= 2:
        valid_indices = [i for i, pos in enumerate(interpolated_positions) if pos is not None]
        if len(valid_indices) >= 2:
            pos1 = interpolated_positions[valid_indices[0]]
            pos2 = interpolated_positions[valid_indices[1]]
            dt_frames = segment_frames[valid_indices[1]] - segment_frames[valid_indices[0]]
            vx_init = (pos2[0] - pos1[0]) / dt_frames
            vy_init = (pos2[1] - pos1[1]) / dt_frames
            tracker.state[2:] = [vx_init, vy_init]
            print(f"Set initial velocity: vx={vx_init:.2f}, vy={vy_init:.2f}")
    
    for i, pos in enumerate(interpolated_positions):
        measurement = np.array(pos) if pos is not None else None
        tracker.predict()
        tracker.update(measurement)
    
    smoothed_positions = np.array(tracker.history)
    confidence_scores = tracker.confidence
    print(f"Generated {len(smoothed_positions)} smoothed positions")
    
    if len(smoothed_positions) >= 7:
        window_length = min(7, len(smoothed_positions))
        if window_length % 2 == 0:
            window_length -= 1
        print(f"Applying Savitzky-Golay filter with window length {window_length}")
        if window_length >= 3:
            smoothed_positions[:, 0] = savgol_filter(smoothed_positions[:, 0], window_length=window_length, polyorder=2)
            smoothed_positions[:, 1] = savgol_filter(smoothed_positions[:, 1], window_length=window_length, polyorder=2)
            print("Savitzky-Golay filter applied to smooth trajectory")
    
    return smoothed_positions, confidence_scores

def correct_bounces_with_table(smoothed_positions, table_coords):
    """Corrects the trajectory by aligning suspected bounce points with the table's y-coordinates."""
    print("Correcting trajectory for pitches and bounces using table coordinates...")
    corrected_positions = smoothed_positions.copy()
    
    top_y = table_coords[1]
    bottom_y = table_coords[3]
    print(f"Table coordinates: top_y={top_y}, bottom_y={bottom_y}")
    
    if len(corrected_positions) < 3:
        print("Too few positions to correct bounces, returning unchanged")
        return corrected_positions
    
    vy = np.gradient(corrected_positions[:, 1])
    bounce_count = 0
    
    for i in range(1, len(corrected_positions) - 1):
        if (vy[i-1] * vy[i+1] < 0):
            if abs(corrected_positions[i, 1] - top_y) < 15 or abs(corrected_positions[i, 1] - bottom_y) < 15:
                if abs(corrected_positions[i, 1] - top_y) < abs(corrected_positions[i, 1] - bottom_y):
                    corrected_positions[i, 1] = top_y
                    print(f"Corrected bounce at frame {i} to table top (y={top_y})")
                else:
                    corrected_positions[i, 1] = bottom_y
                    print(f"Corrected bounce at frame {i} to table bottom (y={bottom_y})")
                bounce_count += 1
    
    print(f"Corrected {bounce_count} bounce points")
    return corrected_positions

def plot_segment_trajectory(segment_frames, valid_positions, valid_frames, smoothed_positions, 
                          segment_label, start_frame, end_frame, save_plot=True, table_coords=None):
    """Create comprehensive trajectory visualization."""
    print(f"Creating trajectory plot for segment {segment_label}...")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    if len(valid_positions) > 0:
        plt.scatter(valid_positions[:, 0], valid_positions[:, 1], alpha=0.6, s=30, color='lightblue', label='Raw detections', zorder=3)
        print(f"Plotted {len(valid_positions)} raw detections")
    
    plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], 'red', linewidth=3, alpha=0.8, label='Smoothed trajectory', zorder=4)
    print("Plotted smoothed trajectory")
    
    if len(smoothed_positions) > 0:
        plt.scatter(smoothed_positions[0, 0], smoothed_positions[0, 1], color='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(smoothed_positions[-1, 0], smoothed_positions[-1, 1], color='red', s=100, marker='s', label='End', zorder=5)
        print("Plotted start and end points")
    
    if table_coords:
        top_y = table_coords[1]
        bottom_y = table_coords[3]
        plt.axhline(y=top_y, color='black', linestyle='--', label='Table Top', zorder=2)
        plt.axhline(y=bottom_y, color='black', linestyle='--', label='Table Bottom', zorder=2)
        print("Added table boundary lines to plot")

    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(f'Trajectory - Segment {segment_label}\n(Frames {start_frame}-{end_frame})')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(segment_frames, smoothed_positions[:, 0], 'blue', linewidth=2, label='Smoothed X')
    if len(valid_positions) > 0:
        plt.scatter(valid_frames, valid_positions[:, 0], alpha=0.6, color='lightblue', s=20, label='Raw X')
    plt.xlabel('Frame Number')
    plt.ylabel('X Position (pixels)')
    plt.title('Horizontal Movement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("Plotted horizontal movement")
    
    plt.subplot(2, 3, 3)
    plt.plot(segment_frames, smoothed_positions[:, 1], 'green', linewidth=2, label='Smoothed Y')
    if len(valid_positions) > 0:
        plt.scatter(valid_frames, valid_positions[:, 1], alpha=0.6, color='lightgreen', s=20, label='Raw Y')
    
    if table_coords:
        top_y = table_coords[1]
        bottom_y = table_coords[3]
        plt.axhline(y=top_y, color='black', linestyle='--', label='Table Top', zorder=2)
        plt.axhline(y=bottom_y, color='black', linestyle='--', label='Table Bottom', zorder=2)
        
    plt.xlabel('Frame Number')
    plt.ylabel('Y Position (pixels)')
    plt.title('Vertical Movement')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("Plotted vertical movement")
    
    plt.subplot(2, 3, 4)
    if len(smoothed_positions) >= 2:
        vx = np.gradient(smoothed_positions[:, 0])
        vy = np.gradient(smoothed_positions[:, 1])
        speed = np.sqrt(vx**2 + vy**2)
        plt.plot(segment_frames, speed, 'purple', linewidth=2)
        plt.xlabel('Frame Number')
        plt.ylabel('Speed (pixels/frame)')
        plt.title('Ball Speed Over Time')
        plt.grid(True, alpha=0.3)
        print("Plotted speed over time")
    
    plt.subplot(2, 3, 5)
    if len(smoothed_positions) >= 2:
        vx = np.gradient(smoothed_positions[:, 0])
        vy = np.gradient(smoothed_positions[:, 1])
        speed = np.sqrt(vx**2 + vy**2)
        if np.max(speed) > 0:
            normalized_speed = speed / np.max(speed)
            colors = plt.cm.plasma(normalized_speed)
            for i in range(len(smoothed_positions) - 1):
                plt.plot([smoothed_positions[i, 0], smoothed_positions[i+1, 0]], 
                        [smoothed_positions[i, 1], smoothed_positions[i+1, 1]], 
                        color=colors[i], linewidth=3, alpha=0.7)
    
    if table_coords:
        top_y = table_coords[1]
        bottom_y = table_coords[3]
        plt.axhline(y=top_y, color='black', linestyle='--', zorder=2)
        plt.axhline(y=bottom_y, color='black', linestyle='--', zorder=2)
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Speed-Colored Trajectory')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    print("Plotted speed-colored trajectory")
    
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    if len(valid_positions) >= 2:
        distances = np.sqrt(np.sum(np.diff(valid_positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        avg_speed = np.mean(distances)
        max_speed = np.max(distances)
        
        stats_text = f"""Trajectory Statistics:
        
Total Frames: {len(segment_frames)}
Valid Detections: {len(valid_positions)}
Coverage: {len(valid_positions)/len(segment_frames)*100:.1f}%

Movement Analysis:
Total Distance: {total_distance:.0f} px
Average Speed: {avg_speed:.2f} px/frame
Max Speed: {max_speed:.2f} px/frame

Bounds:
X Range: {np.min(valid_positions[:, 0]):.0f} - {np.max(valid_positions[:, 0]):.0f}
Y Range: {np.min(valid_positions[:, 1]):.0f} - {np.max(valid_positions[:, 1]):.0f}"""
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        print("Added trajectory statistics to plot")
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("trajectory_segments", exist_ok=True)
        filename = f"trajectory_segments/segment_{segment_label}_frames_{start_frame}_{end_frame}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved as: {filename}")
    
    # Comment out plt.show() to avoid blocking
    # plt.show()
    print("Skipped interactive plot display to avoid blocking")
    
    # Close the figure to free memory
    plt.close()
    print("Closed matplotlib figure")
    
    return smoothed_positions

def process_segment_trajectory(json_filename, start_frame, end_frame, segment_label, 
                             table_coords=None, max_interpolation_gap=5, save_plot=True, show_analysis=True):
    """Main function to process trajectory for a specific segment."""
    
    print(f"\n🏓 Processing Trajectory for Segment {segment_label}")
    print("=" * 60)
    
    try:
        data = load_ball_markup_data(json_filename)
        segment_frames, segment_positions, valid_frames, valid_positions = extract_segment_data(
            data, start_frame, end_frame, segment_label)
        
        if len(valid_positions) == 0:
            print("❌ No valid ball positions found in this segment!")
            return None
        
        interpolated_positions = interpolate_missing_frames(
            segment_frames, segment_positions, valid_frames, valid_positions, max_gap=max_interpolation_gap)
        
        smoothed_positions, confidence_scores = smooth_trajectory_kalman(
            segment_frames, interpolated_positions, valid_positions)
        
        if table_coords:
            smoothed_positions = correct_bounces_with_table(smoothed_positions, table_coords)
        
        result_positions = plot_segment_trajectory(
            segment_frames, valid_positions, valid_frames, smoothed_positions,
            segment_label, start_frame, end_frame, save_plot=save_plot, table_coords=table_coords)
        
        if show_analysis:
            print(f"\n📊 Segment {segment_label} Analysis Complete:")
            print(f"   Frames processed: {start_frame} to {end_frame} ({len(segment_frames)} frames)")
            print(f"   Valid detections: {len(valid_positions)} ({len(valid_positions)/len(segment_frames)*100:.1f}%)")
            print(f"   Average confidence: {np.mean(confidence_scores):.2f}")
        
        print("Trajectory processing completed successfully")
        return {
            'segment_label': segment_label,
            'frames': segment_frames,
            'raw_positions': valid_positions,
            'smoothed_positions': smoothed_positions,
            'confidence': confidence_scores,
            'valid_frames': valid_frames
        }
        
    except Exception as e:
        print(f"❌ Error processing segment {segment_label}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def overlay_trajectory_on_video_segment(video_path, output_path, smoothed_positions, start_frame, end_frame, table_coords=None):
    """
    Overlays a smoothed trajectory path onto a specific video segment.
    """
    print(f"Starting video overlay for segment: frames {start_frame} to {end_frame}")
    print(f"Loading video from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    print("Video loaded successfully")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"Set video to start frame {start_frame}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: width={frame_width}, height={frame_height}, fps={fps}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Initialized video writer for output: {output_path}")
    
    frame_count = start_frame
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {frame_count}")
            break
        
        current_data_idx = frame_count - start_frame
        print(f"Processing frame {frame_count} (index {current_data_idx})")
        
        for i in range(max(0, current_data_idx - 1), current_data_idx):
            if i + 1 < len(smoothed_positions):
                p1 = tuple(smoothed_positions[i].astype(int))
                p2 = tuple(smoothed_positions[i+1].astype(int))
                cv2.line(frame, p1, p2, (0, 0, 255), 3)
                print(f"Drew trajectory line from {p1} to {p2}")
        
        if current_data_idx < len(smoothed_positions):
            current_pos = tuple(smoothed_positions[current_data_idx].astype(int))
            cv2.circle(frame, current_pos, 8, (0, 255, 0), -1)
            print(f"Drew ball position at {current_pos}")
        
        if table_coords:
            p1_top = (table_coords[0], table_coords[1])
            p2_top = (table_coords[2], table_coords[1])
            p1_bottom = (table_coords[0], table_coords[3])
            p2_bottom = (table_coords[2], table_coords[3])
            cv2.line(frame, p1_top, p2_top, (255, 255, 0), 2)
            cv2.line(frame, p1_bottom, p2_bottom, (255, 255, 0), 2)
            print("Drew table boundary lines")
        
        out.write(frame)
        print(f"Wrote frame {frame_count} to output video")
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video segment with trajectory saved to {output_path}")
    print("Video overlay processing completed")

def main():
    """Main function to run the analysis and video overlay."""
    
    print("Starting main execution")
    
    json_filename = "ball_markup_1.json"
    input_video_file = "game_1.mp4"
    start_frame = 2212
    end_frame = 2273
    segment_label = "s1"
    output_video_file = f"output_segment_{segment_label}.mp4"
    table_coordinates = (204, 346, 956, 443)
    
    print(f"Input JSON: {json_filename}")
    print(f"Input video: {input_video_file}")
    print(f"Processing segment {segment_label}: frames {start_frame} to {end_frame}")
    print(f"Table coordinates: {table_coordinates}")
    
    analysis_result = process_segment_trajectory(
        json_filename=json_filename,
        start_frame=start_frame,
        end_frame=end_frame,
        segment_label=segment_label,
        table_coords=table_coordinates,
        save_plot=True,
        show_analysis=True
    )
    
    if analysis_result:
        print("Trajectory analysis completed, proceeding to video overlay")
        smoothed_positions = analysis_result['smoothed_positions']
        print(f"Smoothed positions shape: {smoothed_positions.shape}")
        
        overlay_trajectory_on_video_segment(
            video_path=input_video_file,
            output_path=output_video_file,
            smoothed_positions=smoothed_positions,
            start_frame=start_frame,
            end_frame=end_frame,
            table_coords=table_coordinates
        )
    else:
        print("Trajectory analysis failed. Cannot create video overlay.")
    
    print("Main execution completed")

if __name__ == "__main__":
    main()