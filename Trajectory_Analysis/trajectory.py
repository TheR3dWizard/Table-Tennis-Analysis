# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
# from typing import Tuple, Optional

# class KalmanBallTracker:
#     """Kalman filter for tracking table tennis ball with physics-based motion model."""
    
#     def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=10.0):
#         """
#         Initialize Kalman filter for 2D ball tracking.
        
#         State vector: [x, y, vx, vy] (position and velocity)
#         """
#         self.dt = dt  # Time step (1 frame)
        
#         # State vector: [x, y, vx, vy]
#         self.state = np.zeros(4)
        
#         # State transition matrix (physics: x = x + vx*dt, vx = vx + ax*dt)
#         self.F = np.array([
#             [1, 0, dt, 0],    # x = x + vx*dt
#             [0, 1, 0, dt],    # y = y + vy*dt  
#             [0, 0, 1, 0],     # vx = vx (constant velocity assumption)
#             [0, 0, 0, 1]      # vy = vy + gravity (handled separately)
#         ])
        
#         # Measurement matrix (we observe position only)
#         self.H = np.array([
#             [1, 0, 0, 0],     # Observe x
#             [0, 1, 0, 0]      # Observe y
#         ])
        
#         # Process noise covariance (physics uncertainty)
#         self.Q = np.eye(4) * process_noise
#         self.Q[2:, 2:] *= 2  # Higher uncertainty in velocity
        
#         # Measurement noise covariance (tracking accuracy)
#         self.R = np.eye(2) * measurement_noise
        
#         # State covariance (initial uncertainty)
#         self.P = np.eye(4) * 100
        
#         # Track history
#         self.history = []
#         self.predictions = []
        
#     def predict(self, gravity=0.1):
#         """Predict next state using physics model."""
#         # Add gravity to y-velocity
#         gravity_effect = np.array([0, 0, 0, gravity])
        
#         # Predict state: x(k+1) = F * x(k) + gravity
#         self.state = self.F @ self.state + gravity_effect
        
#         # Predict covariance: P(k+1) = F * P(k) * F^T + Q
#         self.P = self.F @ self.P @ self.F.T + self.Q
        
#         # Store prediction
#         self.predictions.append(self.state[:2].copy())
        
#         return self.state[:2]  # Return predicted position
    
#     def update(self, measurement):
#         """Update state with new measurement (if available)."""
#         if measurement is None:
#             # No measurement available, just store predicted state
#             self.history.append(self.state[:2].copy())
#             return
        
#         # Kalman gain: K = P * H^T * (H * P * H^T + R)^-1
#         S = self.H @ self.P @ self.H.T + self.R
#         K = self.P @ self.H.T @ np.linalg.inv(S)
        
#         # Update state: x = x + K * (z - H * x)
#         innovation = measurement - self.H @ self.state
#         self.state = self.state + K @ innovation
        
#         # Update covariance: P = (I - K * H) * P
#         I = np.eye(4)
#         self.P = (I - K @ self.H) @ self.P
        
#         # Store updated state
#         self.history.append(self.state[:2].copy())
    
#     def initialize_state(self, first_positions, first_frames):
#         """Initialize state using first few valid positions."""
#         if len(first_positions) >= 2:
#             # Estimate initial velocity from first two points
#             dt_init = first_frames[1] - first_frames[0]
#             vx_init = (first_positions[1][0] - first_positions[0][0]) / dt_init
#             vy_init = (first_positions[1][1] - first_positions[0][1]) / dt_init
            
#             self.state = np.array([first_positions[0][0], first_positions[0][1], 
#                                   vx_init, vy_init])
#         else:
#             self.state = np.array([first_positions[0][0], first_positions[0][1], 0, 0])

# def load_data_for_kalman(filename):
#     """Load data and prepare for Kalman filtering."""
#     with open(filename, "r") as f:
#         data = json.load(f)
    
#     frames = np.array(sorted(int(k) for k in data.keys()))
    
#     # Extract measurements (None for missing data)
#     measurements = []
#     for frame in frames:
#         x, y = data[str(frame)]["x"], data[str(frame)]["y"]
#         if x == -1 or y == -1:
#             measurements.append(None)  # Missing measurement
#         else:
#             measurements.append(np.array([x, y]))
    
#     return frames, measurements

# def track_ball_with_kalman(frames, measurements, gravity=0.15):
#     """Track ball using Kalman filter with physics model."""
    
#     # Find first few valid measurements to initialize
#     valid_measurements = [(i, m) for i, m in enumerate(measurements) if m is not None]
    
#     if len(valid_measurements) < 2:
#         raise ValueError("Need at least 2 valid measurements to initialize tracking")
    
#     # Initialize tracker
#     tracker = KalmanBallTracker(dt=1.0, process_noise=0.5, measurement_noise=5.0)
    
#     # Initialize state with first valid measurements
#     first_indices = [valid_measurements[i][0] for i in range(min(3, len(valid_measurements)))]
#     first_positions = [valid_measurements[i][1] for i in range(min(3, len(valid_measurements)))]
#     first_frames = frames[first_indices]
    
#     tracker.initialize_state(first_positions, first_frames)
    
#     # Track through all frames
#     tracked_positions = []
    
#     for i, measurement in enumerate(measurements):
#         # Predict next position
#         predicted_pos = tracker.predict(gravity=gravity)
        
#         # Update with measurement (if available)
#         tracker.update(measurement)
        
#         # Store the final position estimate
#         tracked_positions.append(tracker.history[-1].copy())
    
#     return np.array(tracked_positions), tracker

# def detect_bounces_from_physics(positions, frames, velocity_threshold=2.0):
#     """Detect bounces using physics: look for sudden upward velocity changes."""
    
#     # Calculate velocity
#     vx = np.gradient(positions[:, 0])
#     vy = np.gradient(positions[:, 1])
    
#     # Smooth velocity to reduce noise
#     vy_smooth = savgol_filter(vy, window_length=11, polyorder=2)
    
#     # Find where vertical velocity changes from downward to upward (bounce signature)
#     bounces = []
    
#     for i in range(10, len(vy_smooth) - 10):
#         # Look for: negative velocity becoming positive (bouncing up)
#         before = np.mean(vy_smooth[i-5:i])    # Velocity before
#         after = np.mean(vy_smooth[i:i+5])     # Velocity after
        
#         # Bounce signature: was going down, now going up, with sufficient magnitude
#         if (before > velocity_threshold and after < -velocity_threshold):
#             # Additional check: must be at a local Y maximum (ball at table level)
#             local_y = positions[i-5:i+6, 1]
#             if positions[i, 1] >= np.percentile(local_y, 75):  # Near the bottom in image coords
#                 bounces.append(i)
    
#     # Remove bounces too close together
#     if len(bounces) > 1:
#         filtered_bounces = [bounces[0]]
#         for bounce in bounces[1:]:
#             if bounce - filtered_bounces[-1] >= 100:  # Minimum 100 frames between bounces
#                 filtered_bounces.append(bounce)
#         bounces = filtered_bounces
    
#     return np.array(bounces)

# def plot_kalman_results(frames, original_coords, kalman_positions, bounces):
#     """Plot comparison between original data and Kalman-filtered results."""
    
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle("Kalman Filter Ball Tracking Analysis", fontsize=16, fontweight='bold')
    
#     # 1. Trajectory comparison
#     # Plot original data (where available)
#     valid_mask = (original_coords[:, 0] != -1) & (original_coords[:, 1] != -1)
#     ax1.scatter(original_coords[valid_mask, 0], original_coords[valid_mask, 1], 
#                alpha=0.4, s=15, color='lightblue', label="Raw detections")
    
#     # Plot Kalman filtered trajectory
#     ax1.plot(kalman_positions[:, 0], kalman_positions[:, 1], 
#              'blue', linewidth=2, label="Kalman filtered")
    
#     # Mark bounces
#     if len(bounces) > 0:
#         ax1.scatter(kalman_positions[bounces, 0], kalman_positions[bounces, 1], 
#                    color='red', s=80, zorder=5, label=f"Bounces ({len(bounces)})")
    
#     ax1.set_xlabel("X Position (pixels)")
#     ax1.set_ylabel("Y Position (pixels)")
#     ax1.set_title("Ball Trajectory (Kalman vs Raw)")
#     ax1.invert_yaxis()
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # 2. Y position over time
#     ax2.plot(frames, kalman_positions[:, 1], 'green', linewidth=2, label="Y position")
#     if len(bounces) > 0:
#         ax2.scatter(frames[bounces], kalman_positions[bounces, 1], 
#                    color='red', s=60, zorder=5)
#     ax2.set_xlabel("Frame")
#     ax2.set_ylabel("Y Position (pixels)")
#     ax2.set_title("Height Over Time")
#     ax2.invert_yaxis()
#     ax2.grid(True, alpha=0.3)
    
#     # 3. Velocity components
#     vx = np.gradient(kalman_positions[:, 0])
#     vy = np.gradient(kalman_positions[:, 1])
    
#     ax3.plot(frames, vx, 'blue', linewidth=1.5, label="X velocity")
#     ax3.plot(frames, vy, 'red', linewidth=1.5, label="Y velocity")
#     ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
#     if len(bounces) > 0:
#         ax3.scatter(frames[bounces], vx[bounces], color='blue', s=40, alpha=0.7)
#         ax3.scatter(frames[bounces], vy[bounces], color='red', s=40, alpha=0.7)
#     ax3.set_xlabel("Frame")
#     ax3.set_ylabel("Velocity (pixels/frame)")
#     ax3.set_title("Ball Velocity Components")
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
    
#     # 4. Speed profile
#     speed = np.sqrt(vx**2 + vy**2)
#     ax4.plot(frames, speed, 'purple', linewidth=2, label="Speed")
#     if len(bounces) > 0:
#         ax4.scatter(frames[bounces], speed[bounces], color='red', s=60, zorder=5)
#     ax4.set_xlabel("Frame")
#     ax4.set_ylabel("Speed (pixels/frame)")
#     ax4.set_title("Ball Speed")
#     ax4.legend()
#     ax4.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()

# def analyze_ball_trajectory():
#     """Main analysis function using Kalman filtering."""
    
#     print("🏓 Ball Trajectory Analysis with Kalman Filter")
#     print("=" * 50)
    
#     # Load data
#     frames, measurements = load_data_for_kalman("ball_markup_1.json")
#     original_coords = np.array([[m[0], m[1]] if m is not None else [-1, -1] for m in measurements])
    
#     print(f"Loaded {len(frames)} frames")
#     missing_count = sum(1 for m in measurements if m is None)
#     print(f"Missing measurements: {missing_count}")
    
#     # Track with Kalman filter
#     print("Applying Kalman filter (physics-based tracking)...")
#     kalman_positions, tracker = track_ball_with_kalman(frames, measurements)
    
#     # Detect bounces using physics
#     bounces = detect_bounces_from_physics(kalman_positions, frames)
    
#     # Calculate statistics
#     vx = np.gradient(kalman_positions[:, 0])
#     vy = np.gradient(kalman_positions[:, 1])
#     speed = np.sqrt(vx**2 + vy**2)
    
#     total_distance = np.sum(np.sqrt(np.diff(kalman_positions[:, 0])**2 + 
#                                    np.diff(kalman_positions[:, 1])**2))
    
#     # Print results
#     print(f"\n📊 RESULTS:")
#     print(f"   Trajectory length: {total_distance:.0f} pixels")
#     print(f"   Average speed: {np.mean(speed):.1f} px/frame")
#     print(f"   Maximum speed: {np.max(speed):.1f} px/frame")
#     print(f"   Bounces detected: {len(bounces)}")
    
#     if len(bounces) > 0:
#         bounce_frames = frames[bounces]
#         print(f"   Bounce frames: {bounce_frames}")
        
#         if len(bounces) > 1:
#             intervals = np.diff(bounce_frames)
#             print(f"   Bounce intervals: {intervals} frames")
#             print(f"   Average interval: {np.mean(intervals):.0f} frames")
    
#     # Visualize results
#     plot_kalman_results(frames, original_coords, kalman_positions, bounces)
    
#     return kalman_positions, bounces

# # Alternative simplified version using basic physics
# def simple_physics_interpolation(frames, measurements):
#     """Simple physics-based interpolation without full Kalman filter."""
    
#     # Find valid measurements
#     valid_indices = [i for i, m in enumerate(measurements) if m is not None]
#     valid_positions = [measurements[i] for i in valid_indices]
#     valid_frames = frames[valid_indices]
    
#     if len(valid_positions) < 2:
#         raise ValueError("Need at least 2 valid positions")
    
#     # Initialize output array
#     interpolated = np.zeros((len(frames), 2))
    
#     # Fill in known positions
#     for i, pos in zip(valid_indices, valid_positions):
#         interpolated[i] = pos
    
#     # Interpolate missing segments using physics
#     for i in range(len(valid_indices) - 1):
#         start_idx = valid_indices[i]
#         end_idx = valid_indices[i + 1]
        
#         if end_idx - start_idx <= 1:
#             continue  # No gap to fill
        
#         # Get start and end positions
#         start_pos = valid_positions[i]
#         end_pos = valid_positions[i + 1]
#         dt_total = end_idx - start_idx
        
#         # Estimate constant velocity for this segment
#         vx = (end_pos[0] - start_pos[0]) / dt_total
#         vy = (end_pos[1] - start_pos[1]) / dt_total
        
#         # Fill intermediate positions
#         for j in range(start_idx + 1, end_idx):
#             dt = j - start_idx
#             # Simple physics: x = x0 + v*t
#             interpolated[j, 0] = start_pos[0] + vx * dt
#             interpolated[j, 1] = start_pos[1] + vy * dt
    
#     return interpolated

# def simple_analysis():
#     """Simplified version without full Kalman complexity."""
    
#     print("🏓 Simple Physics-Based Ball Analysis")
#     print("=" * 40)
    
#     # Load data
#     frames, measurements = load_data_for_kalman("ball_markup_1.json")
#     print(f"Loaded {len(frames)} frames")
    
#     # Physics-based interpolation
#     positions = simple_physics_interpolation(frames, measurements)
#     print("Applied physics-based interpolation")
    
#     # Light smoothing
#     window = min(9, len(frames) // 10)
#     if window % 2 == 0:
#         window += 1
#     if window >= 3:
#         x_smooth = savgol_filter(positions[:, 0], window_length=window, polyorder=2)
#         y_smooth = savgol_filter(positions[:, 1], window_length=window, polyorder=2)
#         positions = np.column_stack([x_smooth, y_smooth])
    
#     # Simple bounce detection
#     bounces = []
#     y_coords = positions[:, 1]
    
#     # Look for clear local maxima in Y (remember: Y increases downward in image coords)
#     for i in range(20, len(y_coords) - 20):
#         window = y_coords[i-10:i+11]
#         if (y_coords[i] == np.max(window) and 
#             np.max(window) - np.min(window) > 15):  # Significant bounce
#             bounces.append(i)
    
#     # Filter bounces
#     if len(bounces) > 1:
#         filtered = [bounces[0]]
#         for b in bounces[1:]:
#             if b - filtered[-1] >= 50:  # Minimum separation
#                 filtered.append(b)
#         bounces = filtered
    
#     bounces = np.array(bounces)
    
#     # Simple plot
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label="Ball path")
#     if len(bounces) > 0:
#         plt.scatter(positions[bounces, 0], positions[bounces, 1], 
#                    color='red', s=80, label=f"Bounces ({len(bounces)})")
#     plt.xlabel("X (pixels)")
#     plt.ylabel("Y (pixels)")
#     plt.title("Ball Trajectory")
#     plt.gca().invert_yaxis()
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(1, 2, 2)
#     plt.plot(frames, positions[:, 1], 'g-', linewidth=2)
#     if len(bounces) > 0:
#         plt.scatter(frames[bounces], positions[bounces, 1], color='red', s=60)
#     plt.xlabel("Frame")
#     plt.ylabel("Y Position (pixels)")
#     plt.title("Height Over Time")
#     plt.gca().invert_yaxis()
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print results
#     print(f"\nResults:")
#     print(f"   Bounces detected: {len(bounces)}")
#     if len(bounces) > 0:
#         print(f"   Bounce frames: {frames[bounces]}")

# def main():
#     """Choose analysis method."""
#     print("Choose analysis method:")
#     print("1. Full Kalman Filter (advanced)")
#     print("2. Simple Physics Interpolation (recommended)")
    
#     choice = input("Enter choice (1 or 2): ").strip()
    
#     if choice == "1":
#         analyze_ball_trajectory()
#     else:
#         simple_analysis()

# if __name__ == "__main__":
#     main()






import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

class SimpleKalmanTracker:
    """Simplified Kalman filter for 2D ball tracking."""
    def __init__(self, dt=1.0, process_noise=0.5, measurement_noise=5.0):
        self.dt = dt
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 100
        self.history = []

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.history.append(self.state[:2].copy())

    def update(self, measurement):
        if measurement is None:
            return
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P

def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    frames = np.array(sorted(int(k) for k in data.keys()))
    measurements = []
    for frame in frames:
        x, y = data[str(frame)]["x"], data[str(frame)]["y"]
        measurements.append(None if x == -1 or y == -1 else np.array([x, y]))
    return frames, measurements

def interpolate_gaps(frames, measurements, max_gap=5):
    """Interpolate short gaps (<= max_gap frames) with linear interpolation."""
    interpolated = measurements.copy()
    for i in range(len(measurements)):
        if measurements[i] is None:
            prev_idx = next((j for j in range(i-1, -1, -1) if measurements[j] is not None), None)
            next_idx = next((j for j in range(i+1, len(measurements)) if measurements[j] is not None), None)
            if prev_idx is not None and next_idx is not None and (next_idx - prev_idx) <= max_gap:
                prev_pos = measurements[prev_idx]
                next_pos = measurements[next_idx]
                t = (frames[i] - frames[prev_idx]) / (frames[next_idx] - frames[prev_idx])
                interpolated[i] = prev_pos + t * (next_pos - prev_pos)
    return interpolated

def segment_trajectory(frames, measurements, gap_threshold=5):
    """Split trajectory into segments based on large gaps."""
    segments = []
    start_idx = 0
    for i in range(1, len(measurements)):
        if measurements[i] is None and (i == len(measurements)-1 or measurements[i+1] is None):
            if i - start_idx > 1:
                segments.append((start_idx, i))
            start_idx = i + 1
    if start_idx < len(measurements):
        segments.append((start_idx, len(measurements)))
    return segments

def process_trajectory(filename):
    """Process and plot smoothed trajectory segments."""
    # Ensure output directory exists
    os.makedirs("trajectory", exist_ok=True)
    
    # Load and preprocess data
    frames, measurements = load_data(filename)
    measurements = interpolate_gaps(frames, measurements, max_gap=5)
    
    # Segment trajectory
    segments = segment_trajectory(frames, measurements, gap_threshold=5)
    
    for seg_idx, (start, end) in enumerate(segments):
        seg_frames = frames[start:end]
        seg_measurements = measurements[start:end]
        
        # Skip empty segments
        if not any(m is not None for m in seg_measurements):
            continue
        
        # Initialize Kalman filter
        tracker = SimpleKalmanTracker()
        valid_measurements = [m for m in seg_measurements if m is not None]
        if not valid_measurements:
            continue
        tracker.state[:2] = valid_measurements[0]  # Initialize with first valid point
        
        # Process segment with Kalman filter
        smoothed_positions = []
        for measurement in seg_measurements:
            tracker.predict()
            tracker.update(measurement)
            smoothed_positions.append(tracker.history[-1].copy())
        smoothed_positions = np.array(smoothed_positions)
        
        # Apply light smoothing
        if len(smoothed_positions) >= 7:
            smoothed_positions[:, 0] = savgol_filter(smoothed_positions[:, 0], window_length=7, polyorder=2)
            smoothed_positions[:, 1] = savgol_filter(smoothed_positions[:, 1], window_length=7, polyorder=2)
        
        # Plot segment
        plt.figure(figsize=(8, 6))
        valid_mask = [m is not None for m in seg_measurements]
        valid_positions = np.array([m for m in seg_measurements if m is not None])
        if len(valid_positions) > 0:
            plt.scatter(valid_positions[:, 0], valid_positions[:, 1], alpha=0.4, s=15, color='lightblue', label='Raw')
        plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], 'blue', linewidth=2, label='Smoothed')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.title(f'Trajectory Segment {seg_idx + 1} (Frames {seg_frames[0]}-{seg_frames[-1]})')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_file = f"trajectory/ball_markup_segment_{seg_idx + 1}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    process_trajectory("ball_markup_1.json")