import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, UnivariateSpline, splprep, splev
import os

class EnhancedKalmanTracker:
    """Enhanced Kalman filter optimized for smooth ball tracking."""
    def __init__(self, dt=1.0, process_noise=0.1, measurement_noise=2.0, velocity_decay=0.98):
        self.dt = dt
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.velocity_decay = velocity_decay
        
        # State transition matrix with velocity decay
        self.F = np.array([
            [1, 0, dt, 0], 
            [0, 1, 0, dt], 
            [0, 0, velocity_decay, 0], 
            [0, 0, 0, velocity_decay]
        ])
        
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise  # Reduced process noise
        self.R = np.eye(2) * measurement_noise  # Reduced measurement noise
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
    print(f"Successfully loaded JSON data with {len(data)} frames")
    return data

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
            
            if x != -1 and y != -1:
                segment_positions.append([x, y])
                valid_frames.append(frame_num)
            else:
                segment_positions.append(None)
        else:
            segment_positions.append(None)
    
    valid_positions = np.array([pos for pos in segment_positions if pos is not None])
    print(f"Found {len(valid_positions)} valid ball positions out of {len(segment_frames)} frames")
    
    return np.array(segment_frames), segment_positions, np.array(valid_frames), valid_positions

def interpolate_missing_frames(segment_frames, segment_positions, valid_frames, valid_positions, max_gap=8):
    """Enhanced interpolation using cubic splines."""
    print(f"Starting enhanced interpolation with max gap {max_gap}")
    
    if len(valid_positions) < 4:
        print("Warning: Less than 4 valid points, using linear interpolation")
        if len(valid_positions) >= 2:
            interp_x = interp1d(valid_frames, valid_positions[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_y = interp1d(valid_frames, valid_positions[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        else:
            return segment_positions
    else:
        # Use cubic spline for smoother interpolation
        interp_x = interp1d(valid_frames, valid_positions[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_y = interp1d(valid_frames, valid_positions[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    interpolated_positions = segment_positions.copy()
    interpolated_count = 0
    
    for i, frame_num in enumerate(segment_frames):
        if segment_positions[i] is None:
            # Find nearest valid points
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
            
            # Interpolate if gap is within threshold
            if prev_valid is not None and next_valid is not None:
                if (next_valid - prev_valid) <= max_gap:
                    x_interp = float(interp_x(frame_num))
                    y_interp = float(interp_y(frame_num))
                    interpolated_positions[i] = [x_interp, y_interp]
                    interpolated_count += 1
    
    print(f"Interpolated {interpolated_count} missing frames")
    return interpolated_positions

def create_smooth_spline_trajectory(segment_frames, interpolated_positions, smoothing_factor=None):
    """Create ultra-smooth trajectory using B-splines."""
    print("Creating smooth spline trajectory...")
    
    # Filter out None positions
    valid_indices = [i for i, pos in enumerate(interpolated_positions) if pos is not None]
    if len(valid_indices) < 4:
        print("Not enough points for spline fitting")
        return np.array([pos for pos in interpolated_positions if pos is not None])
    
    valid_positions = np.array([interpolated_positions[i] for i in valid_indices])
    valid_frame_indices = np.array([segment_frames[i] for i in valid_indices])
    
    # Normalize frame indices for better spline fitting
    normalized_frames = (valid_frame_indices - valid_frame_indices[0]) / (valid_frame_indices[-1] - valid_frame_indices[0])
    
    # Automatic smoothing factor based on data density
    if smoothing_factor is None:
        smoothing_factor = len(valid_positions) * 0.5
    
    try:
        # Fit parametric spline
        tck, u = splprep([valid_positions[:, 0], valid_positions[:, 1]], s=smoothing_factor, k=3)
        
        # Generate high-resolution spline points
        u_fine = np.linspace(0, 1, len(segment_frames) * 3)  # 3x resolution for smoother curves
        spline_points = splev(u_fine, tck)
        spline_trajectory = np.column_stack(spline_points)
        
        print(f"Created spline trajectory with {len(spline_trajectory)} points")
        return spline_trajectory
        
    except Exception as e:
        print(f"Spline fitting failed: {e}, falling back to Kalman smoothing")
        return smooth_trajectory_kalman(segment_frames, interpolated_positions, valid_positions)[0]

def smooth_trajectory_kalman(segment_frames, interpolated_positions, valid_positions):
    """Enhanced Kalman filtering for trajectory smoothing."""
    print("Applying enhanced Kalman filter...")
    
    tracker = EnhancedKalmanTracker(dt=1.0, process_noise=0.05, measurement_noise=1.5, velocity_decay=0.99)
    
    # Initialize with first valid position
    first_valid_pos = next(pos for pos in interpolated_positions if pos is not None)
    tracker.state[:2] = first_valid_pos
    
    # Initialize velocity if possible
    if len(valid_positions) >= 2:
        valid_indices = [i for i, pos in enumerate(interpolated_positions) if pos is not None]
        if len(valid_indices) >= 2:
            pos1 = interpolated_positions[valid_indices[0]]
            pos2 = interpolated_positions[valid_indices[1]]
            dt_frames = segment_frames[valid_indices[1]] - segment_frames[valid_indices[0]]
            vx_init = (pos2[0] - pos1[0]) / dt_frames
            vy_init = (pos2[1] - pos1[1]) / dt_frames
            tracker.state[2:] = [vx_init, vy_init]
    
    # Process all positions
    for i, pos in enumerate(interpolated_positions):
        measurement = np.array(pos) if pos is not None else None
        tracker.predict()
        tracker.update(measurement)
    
    smoothed_positions = np.array(tracker.history)
    confidence_scores = tracker.confidence
    
    # Apply multi-pass Savitzky-Golay smoothing for ultra-smooth curves
    if len(smoothed_positions) >= 9:
        # First pass with larger window
        window_length = min(9, len(smoothed_positions))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 5:
            smoothed_positions[:, 0] = savgol_filter(smoothed_positions[:, 0], window_length=window_length, polyorder=3)
            smoothed_positions[:, 1] = savgol_filter(smoothed_positions[:, 1], window_length=window_length, polyorder=3)
            
        # Second pass with smaller window for fine-tuning
        if len(smoothed_positions) >= 7:
            window_length = min(7, len(smoothed_positions))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 5:
                smoothed_positions[:, 0] = savgol_filter(smoothed_positions[:, 0], window_length=window_length, polyorder=2)
                smoothed_positions[:, 1] = savgol_filter(smoothed_positions[:, 1], window_length=window_length, polyorder=2)
    
    print(f"Generated {len(smoothed_positions)} ultra-smooth positions")
    return smoothed_positions, confidence_scores

def draw_smooth_curve(frame, points, color=(0, 0, 255), thickness=3):
    """Draw smooth curves using multiple small line segments."""
    if len(points) < 2:
        return
    
    # Convert to integer points
    points = points.astype(np.int32)
    
    # Draw curve using small connected segments for smooth appearance
    for i in range(len(points) - 1):
        cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), color, thickness)

def draw_bezier_curve(frame, control_points, color=(0, 0, 255), thickness=3, num_points=50):
    """Draw smooth Bezier curves between control points."""
    if len(control_points) < 4:
        # Fallback to simple line drawing
        draw_smooth_curve(frame, control_points, color, thickness)
        return
    
    # Generate smooth curve points using Bezier interpolation
    curve_points = []
    for i in range(0, len(control_points) - 3, 3):
        # Take 4 points at a time for cubic Bezier
        p0, p1, p2, p3 = control_points[i:i+4]
        
        for t in np.linspace(0, 1, num_points // ((len(control_points) - 1) // 3)):
            # Cubic Bezier formula
            point = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
            curve_points.append(point.astype(int))
    
    # Draw the smooth curve
    curve_points = np.array(curve_points)
    draw_smooth_curve(frame, curve_points, color, thickness)

def overlay_trajectory_on_video_segment(video_path, output_path, smoothed_positions, start_frame, end_frame, 
                                      use_spline_curve=True, trail_length=15):
    """
    Enhanced video overlay with smooth curved trajectories.
    """
    print(f"Starting enhanced video overlay for segment: frames {start_frame} to {end_frame}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create high-resolution curve points for smooth visualization
    if use_spline_curve and len(smoothed_positions) > 4:
        try:
            # Create parametric spline for ultra-smooth curves
            tck, u = splprep([smoothed_positions[:, 0], smoothed_positions[:, 1]], s=len(smoothed_positions)*0.3, k=3)
            u_fine = np.linspace(0, 1, len(smoothed_positions) * 5)  # 5x resolution
            spline_x, spline_y = splev(u_fine, tck)
            curve_points = np.column_stack([spline_x, spline_y])
            print(f"Created ultra-smooth spline curve with {len(curve_points)} points")
        except:
            curve_points = smoothed_positions
            print("Spline creation failed, using original smoothed points")
    else:
        curve_points = smoothed_positions
    
    frame_count = start_frame
    while frame_count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_data_idx = frame_count - start_frame
        
        # Draw full trajectory as smooth curve (red)
        draw_smooth_curve(frame, curve_points, color=(0, 0, 255), thickness=3)
        
        # Draw trailing path (fading blue)
        if current_data_idx < len(smoothed_positions) and trail_length > 0:
            start_trail = max(0, current_data_idx - trail_length)
            trail_points = smoothed_positions[start_trail:current_data_idx + 1]
            
            if len(trail_points) > 1:
                # Draw fading trail
                for i in range(len(trail_points) - 1):
                    alpha = (i + 1) / len(trail_points)  # Fading effect
                    color_intensity = int(255 * alpha)
                    cv2.line(frame, 
                           tuple(trail_points[i].astype(int)), 
                           tuple(trail_points[i + 1].astype(int)), 
                           (color_intensity, color_intensity, 0), 2)
        
        # Ball marking removed
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Enhanced video with curved trajectory saved to {output_path}")

def process_segment_trajectory(json_filename, start_frame, end_frame, segment_label, 
                             max_interpolation_gap=8, save_plot=True, show_analysis=True,
                             use_spline=True):
    """Main function to process trajectory with enhanced smoothing."""
    
    print(f"\n🏓 Processing Enhanced Trajectory for Segment {segment_label}")
    print("=" * 60)
    
    try:
        data = load_ball_markup_data(json_filename)
        segment_frames, segment_positions, valid_frames, valid_positions = extract_segment_data(
            data, start_frame, end_frame, segment_label)
        
        if len(valid_positions) == 0:
            print("❌ No valid ball positions found in this segment!")
            return None
        
        # Enhanced interpolation
        interpolated_positions = interpolate_missing_frames(
            segment_frames, segment_positions, valid_frames, valid_positions, max_gap=max_interpolation_gap)
        
        # Choose smoothing method
        if use_spline and len(valid_positions) >= 6:
            print("Using spline-based smoothing for ultra-smooth curves")
            smoothed_positions = create_smooth_spline_trajectory(segment_frames, interpolated_positions)
            confidence_scores = [1.0] * len(smoothed_positions)
        else:
            print("Using enhanced Kalman filtering")
            smoothed_positions, confidence_scores = smooth_trajectory_kalman(
                segment_frames, interpolated_positions, valid_positions)
        
        # Create visualization
        if save_plot:
            plt.figure(figsize=(12, 8))
            
            # Plot raw detections
            if len(valid_positions) > 0:
                plt.scatter(valid_positions[:, 0], valid_positions[:, 1], 
                          alpha=0.6, color='lightblue', s=30, label='Raw detections')
            
            # Plot smooth trajectory
            plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], 
                    'red', linewidth=3, alpha=0.8, label='Smooth trajectory')
            
            # Mark start and end
            if len(smoothed_positions) > 0:
                plt.scatter(smoothed_positions[0, 0], smoothed_positions[0, 1], 
                          color='green', s=100, marker='o', label='Start', zorder=5)
                plt.scatter(smoothed_positions[-1, 0], smoothed_positions[-1, 1], 
                          color='red', s=100, marker='s', label='End', zorder=5)
            
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            plt.title(f'Enhanced Smooth Trajectory - Segment {segment_label}\n(Frames {start_frame}-{end_frame})')
            plt.legend()
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            
            os.makedirs("trajectory_segments", exist_ok=True)
            filename = f"trajectory_segments/enhanced_segment_{segment_label}_frames_{start_frame}_{end_frame}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Enhanced trajectory plot saved as: {filename}")
        
        if show_analysis:
            print(f"\n📊 Enhanced Segment {segment_label} Analysis Complete:")
            print(f"   Frames processed: {start_frame} to {end_frame} ({len(segment_frames)} frames)")
            print(f"   Valid detections: {len(valid_positions)} ({len(valid_positions)/len(segment_frames)*100:.1f}%)")
            print(f"   Smoothing method: {'B-spline' if use_spline and len(valid_positions) >= 6 else 'Enhanced Kalman'}")
            print(f"   Final trajectory points: {len(smoothed_positions)}")
        
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

def main():
    """Main function with enhanced trajectory processing."""
    
    print("Starting Enhanced Trajectory Analysis")
    
    json_filename = "ball_positions.json"
    input_video_file = "game_1.mp4"
    start_frame = 18
    end_frame = 2321
    segment_label = "s7"
    output_video_file = f"final_{segment_label}_{input_video_file}.mp4"
    
    print(f"Processing enhanced segment {segment_label}: frames {start_frame} to {end_frame}")
    
    # Process trajectory with enhanced smoothing
    analysis_result = process_segment_trajectory(
        json_filename=json_filename,
        start_frame=start_frame,
        end_frame=end_frame,
        segment_label=segment_label,
        max_interpolation_gap=10,  # Allow longer gaps for interpolation
        save_plot=True,
        show_analysis=True,
        use_spline=True  # Enable spline-based ultra-smooth curves
    )
    
    if analysis_result:
        print("Enhanced trajectory analysis completed, creating curved video overlay")
        smoothed_positions = analysis_result['smoothed_positions']
        
        overlay_trajectory_on_video_segment(
            video_path=input_video_file,
            output_path=output_video_file,
            smoothed_positions=smoothed_positions,
            start_frame=start_frame,
            end_frame=end_frame,
            use_spline_curve=True,  # Enable ultra-smooth curve rendering
            trail_length=20  # Show trailing effect
        )
        
        print("🎉 Enhanced curved trajectory video created successfully!")
    else:
        print("❌ Enhanced trajectory analysis failed.")

if __name__ == "__main__":
    main()