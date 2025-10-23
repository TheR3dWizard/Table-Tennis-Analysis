import json
import numpy as np
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

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def update(self, measurement):
        if measurement is None:
            self.history.append(self.state[:2].copy())
            self.confidence.append(0.3)
            return
        
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        self.history.append(self.state[:2].copy())
        self.confidence.append(1.0)

def load_ball_markup_data(json_filename):
    """Load ball markup data from JSON file."""
    print(f"Loading ball markup data from {json_filename}...")
    
    with open(json_filename, "r") as f:
        data = json.load(f)
    
    return data

def extract_segment_data(data, start_frame, end_frame, segment_label):
    """Extract trajectory data for a specific segment."""
    print(f"Extracting data for segment {segment_label}: frames {start_frame} to {end_frame}")
    
    segment_frames = []
    segment_positions = []
    valid_frames = []
    
    for frame_num in range(start_frame, end_frame + 1):
        frame_str = str(frame_num)
        
        if frame_str in data:
            x = data[frame_str]["x"]
            y = data[frame_str]["y"]
            
            segment_frames.append(frame_num)
            
            # Check for invalid detections (-1, -1)
            if x != -1 and y != -1:
                segment_positions.append([x, y])
                valid_frames.append(frame_num)
            else:
                segment_positions.append(None)  # Invalid detection
        else:
            # Frame not in data
            segment_frames.append(frame_num)
            segment_positions.append(None)
    
    valid_positions = np.array([pos for pos in segment_positions if pos is not None])
    
    print(f"Found {len(valid_positions)} valid ball positions out of {len(segment_frames)} frames")
    print(f"Coverage: {len(valid_positions)/len(segment_frames)*100:.1f}%")
    
    return np.array(segment_frames), segment_positions, np.array(valid_frames), valid_positions

def interpolate_missing_frames(segment_frames, segment_positions, valid_frames, valid_positions, max_gap=5):
    """Interpolate missing frames using cubic spline if enough points available."""
    if len(valid_positions) < 4:
        print("Warning: Less than 4 valid points, skipping interpolation")
        return segment_positions
    
    print(f"Interpolating missing frames (max gap: {max_gap} frames)...")
    
    # Create interpolation functions
    interp_x = interp1d(valid_frames, valid_positions[:, 0], kind='cubic', 
                       bounds_error=False, fill_value='extrapolate')
    interp_y = interp1d(valid_frames, valid_positions[:, 1], kind='cubic', 
                       bounds_error=False, fill_value='extrapolate')
    
    interpolated_positions = segment_positions.copy()
    
    for i, frame_num in enumerate(segment_frames):
        if segment_positions[i] is None:
            # Check if we should interpolate (gap not too large)
            prev_valid = None
            next_valid = None
            
            # Find previous valid frame
            for j in range(i-1, -1, -1):
                if segment_positions[j] is not None:
                    prev_valid = segment_frames[j]
                    break
            
            # Find next valid frame
            for j in range(i+1, len(segment_frames)):
                if segment_positions[j] is not None:
                    next_valid = segment_frames[j]
                    break
            
            # Interpolate if gap is small enough
            if prev_valid is not None and next_valid is not None:
                if (next_valid - prev_valid) <= max_gap:
                    x_interp = float(interp_x(frame_num))
                    y_interp = float(interp_y(frame_num))
                    interpolated_positions[i] = [x_interp, y_interp]
    
    return interpolated_positions

def smooth_trajectory_kalman(segment_frames, interpolated_positions, valid_positions):
    """Apply Kalman filtering for trajectory smoothing."""
    print("Applying Kalman filter for trajectory smoothing...")
    
    # Initialize tracker
    tracker = SegmentKalmanTracker()
    
    # Set initial state with first valid position
    first_valid_pos = next(pos for pos in interpolated_positions if pos is not None)
    tracker.state[:2] = first_valid_pos
    
    # Estimate initial velocity if we have multiple valid points
    if len(valid_positions) >= 2:
        # Find first two valid positions
        valid_indices = [i for i, pos in enumerate(interpolated_positions) if pos is not None]
        if len(valid_indices) >= 2:
            pos1 = interpolated_positions[valid_indices[0]]
            pos2 = interpolated_positions[valid_indices[1]]
            dt_frames = segment_frames[valid_indices[1]] - segment_frames[valid_indices[0]]
            
            vx_init = (pos2[0] - pos1[0]) / dt_frames
            vy_init = (pos2[1] - pos1[1]) / dt_frames
            tracker.state[2:] = [vx_init, vy_init]
    
    # Process all frames
    for pos in interpolated_positions:
        measurement = np.array(pos) if pos is not None else None
        tracker.predict()
        tracker.update(measurement)
    
    smoothed_positions = np.array(tracker.history)
    confidence_scores = tracker.confidence
    
    # Apply additional Savitzky-Golay smoothing if trajectory is long enough
    if len(smoothed_positions) >= 7:
        window_length = min(7, len(smoothed_positions))
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:
            smoothed_positions[:, 0] = savgol_filter(smoothed_positions[:, 0], 
                                                   window_length=window_length, polyorder=2)
            smoothed_positions[:, 1] = savgol_filter(smoothed_positions[:, 1], 
                                                   window_length=window_length, polyorder=2)
    
    return smoothed_positions, confidence_scores

def correct_bounces_with_table(smoothed_positions, table_coords):
    """
    Corrects the trajectory by aligning suspected bounce points with the table's y-coordinates.
    Args:
        smoothed_positions (np.ndarray): The smoothed trajectory points.
        table_coords (tuple): A tuple (left_x, top_y, right_x, bottom_y) of table coordinates.
    Returns:
        np.ndarray: The bounce-corrected trajectory.
    """
    print("Correcting trajectory for pitches and bounces using table coordinates...")
    corrected_positions = smoothed_positions.copy()
    
    top_y = table_coords[1]
    bottom_y = table_coords[3]
    
    if len(corrected_positions) < 3:
        return corrected_positions
    
    # Calculate vertical velocity
    vy = np.gradient(corrected_positions[:, 1])
    
    for i in range(1, len(corrected_positions) - 1):
        # Check for a change in vertical direction (potential bounce)
        if (vy[i-1] * vy[i+1] < 0):
            # Check if the point is close to the table
            if abs(corrected_positions[i, 1] - top_y) < 15 or abs(corrected_positions[i, 1] - bottom_y) < 15:
                # Snap to the closest table line
                if abs(corrected_positions[i, 1] - top_y) < abs(corrected_positions[i, 1] - bottom_y):
                    corrected_positions[i, 1] = top_y
                else:
                    corrected_positions[i, 1] = bottom_y
                print(f"   -> Bounce detected and corrected at frame {i}")
                
    return corrected_positions

def plot_segment_trajectory(segment_frames, valid_positions, valid_frames, smoothed_positions, 
                          segment_label, start_frame, end_frame, save_plot=True, table_coords=None):
    """Create comprehensive trajectory visualization."""
    print(f"Creating trajectory plot for segment {segment_label}...")
    
    plt.figure(figsize=(15, 10))
    
    # Main trajectory plot
    plt.subplot(2, 3, 1)
    if len(valid_positions) > 0:
        plt.scatter(valid_positions[:, 0], valid_positions[:, 1], 
                   alpha=0.6, s=30, color='lightblue', label='Raw detections', zorder=3)
    
    plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], 
             'red', linewidth=3, alpha=0.8, label='Smoothed trajectory', zorder=4)
    
    # Mark start and end points
    if len(smoothed_positions) > 0:
        plt.scatter(smoothed_positions[0, 0], smoothed_positions[0, 1], 
                   color='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(smoothed_positions[-1, 0], smoothed_positions[-1, 1], 
                   color='red', s=100, marker='s', label='End', zorder=5)
    
    if table_coords:
        top_y = table_coords[1]
        bottom_y = table_coords[3]
        plt.axhline(y=top_y, color='black', linestyle='--', label='Table Top', zorder=2)
        plt.axhline(y=bottom_y, color='black', linestyle='--', label='Table Bottom', zorder=2)

    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(f'Trajectory - Segment {segment_label}\n(Frames {start_frame}-{end_frame})')
    plt.legend()
    plt.gca().invert_yaxis()  # Invert Y-axis for image coordinates
    plt.grid(True, alpha=0.3)
    
    # X position over time
    plt.subplot(2, 3, 2)
    plt.plot(segment_frames, smoothed_positions[:, 0], 'blue', linewidth=2, label='Smoothed X')
    if len(valid_positions) > 0:
        plt.scatter(valid_frames, valid_positions[:, 0], alpha=0.6, color='lightblue', s=20, label='Raw X')
    plt.xlabel('Frame Number')
    plt.ylabel('X Position (pixels)')
    plt.title('Horizontal Movement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y position over time
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
    
    # Speed analysis
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
    
    # Trajectory with speed coloring
    plt.subplot(2, 3, 5)
    if len(smoothed_positions) >= 2:
        vx = np.gradient(smoothed_positions[:, 0])
        vy = np.gradient(smoothed_positions[:, 1])
        speed = np.sqrt(vx**2 + vy**2)
        
        # Normalize speed for coloring
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
    
    # Statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')
    
    # Calculate statistics
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
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("trajectory_segments", exist_ok=True)
        filename = f"trajectory_segments/segment_{segment_label}_frames_{start_frame}_{end_frame}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved as: {filename}")
    
    plt.show()
    
    return smoothed_positions

def process_segment_trajectory(json_filename, start_frame, end_frame, segment_label, 
                             table_coords=None, max_interpolation_gap=5, save_plot=True, show_analysis=True):
    """Main function to process trajectory for a specific segment."""
    
    print(f"\n🏓 Processing Trajectory for Segment {segment_label}")
    print("=" * 60)
    
    try:
        # Load ball markup data
        data = load_ball_markup_data(json_filename)
        
        # Extract segment data
        segment_frames, segment_positions, valid_frames, valid_positions = extract_segment_data(
            data, start_frame, end_frame, segment_label)
        
        if len(valid_positions) == 0:
            print("❌ No valid ball positions found in this segment!")
            return None
        
        # Interpolate missing frames
        interpolated_positions = interpolate_missing_frames(
            segment_frames, segment_positions, valid_frames, valid_positions, 
            max_gap=max_interpolation_gap)
        
        # Smooth trajectory using Kalman filter
        smoothed_positions, confidence_scores = smooth_trajectory_kalman(
            segment_frames, interpolated_positions, valid_positions)
        
        # New step: Correct for pitches/bounces using table coordinates
        if table_coords:
            smoothed_positions = correct_bounces_with_table(smoothed_positions, table_coords)
        
        # Create visualization
        result_positions = plot_segment_trajectory(
            segment_frames, valid_positions, valid_frames, smoothed_positions,
            segment_label, start_frame, end_frame, save_plot=save_plot, table_coords=table_coords)
        
        if show_analysis:
            print(f"\n📊 Segment {segment_label} Analysis Complete:")
            print(f"   Frames processed: {start_frame} to {end_frame} ({len(segment_frames)} frames)")
            print(f"   Valid detections: {len(valid_positions)} ({len(valid_positions)/len(segment_frames)*100:.1f}%)")
            print(f"   Average confidence: {np.mean(confidence_scores):.2f}")
        
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

# Example usage and predefined segments
def main():
    """Example usage with predefined segments."""
    
    # Your predefined segments
    segments = {
        's1': (2212, 2273),
        's2': (2196, 2247),
        's3': (9228, 9308)
    }
    
    # Table coordinates: (left_x, top_y, right_x, bottom_y)
    table_coordinates = (204, 346, 956, 443)
    
    json_filename = "ball_markup_1.json"  # Update with your JSON filename
    
    print("🏓 Ball Trajectory Segment Analyzer")
    print("=" * 50)
    
    # Process each segment
    results = {}
    
    for segment_label, (start_frame, end_frame) in segments.items():
        result = process_segment_trajectory(
            json_filename=json_filename,
            start_frame=start_frame,
            end_frame=end_frame,
            segment_label=segment_label,
            table_coords=table_coordinates,
            max_interpolation_gap=5,
            save_plot=True,
            show_analysis=True
        )
        
        if result:
            results[segment_label] = result
    
    # Summary
    print(f"\n✅ Processing Complete!")
    print(f"Successfully processed {len(results)} out of {len(segments)} segments")
    print("📁 Plots saved in 'trajectory_segments/' directory")
    
    return results

if __name__ == "__main__":
    
    # Table coordinates: (left_x, top_y, right_x, bottom_y)
    table_coordinates = (204, 346, 956, 443)
    
    # Option 1: Process single segment
    result = process_segment_trajectory(
        json_filename="ball_markup_1.json",  # Update with your JSON filename
        start_frame=2212,  # S1 start
        end_frame=2273,    # S1 end
        segment_label="s1",
        table_coords=table_coordinates,
        max_interpolation_gap=5,
        save_plot=True,
        show_analysis=True
    )