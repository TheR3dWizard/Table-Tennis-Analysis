import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import os
import sys

class SegmentKalmanTracker:
    """Enhanced Kalman filter for segment-based ball tracking."""
    def __init__(self, dt=1.0, process_noise=0.8, measurement_noise=2.0):
        self.dt = dt
        self.state = np.zeros(4)  # [x, y, vx, vy]
        
        # State transition matrix with physics
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 0.95, 0],  # Small velocity decay
            [0, 0, 0, 0.95]
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.array([
            [0.1, 0, 0.1, 0],
            [0, 0.1, 0, 0.1], 
            [0.1, 0, process_noise, 0],
            [0, 0.1, 0, process_noise]
        ])
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # Error covariance
        self.P = np.eye(4) * 50
        
        # History tracking
        self.history = []
        self.confidence = []

    def predict(self):
        """Predict next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].copy()

    def update(self, measurement):
        """Update with measurement if available."""
        if measurement is None:
            # No measurement - store prediction with lower confidence
            self.history.append(self.state[:2].copy())
            self.confidence.append(0.2)
            return
        
        # Kalman update equations
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        # Store updated state with high confidence
        self.history.append(self.state[:2].copy())
        self.confidence.append(1.0)

def load_ball_markup_data(json_filename):
    """Load and validate ball markup data from JSON file."""
    print(f"📂 Loading ball markup data from {json_filename}...")
    
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"❌ JSON file not found: {json_filename}")
    
    try:
        with open(json_filename, "r") as f:
            data = json.load(f)
        
        print(f"✅ Successfully loaded data with {len(data)} frame entries")
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON format in {json_filename}: {str(e)}")

def validate_video_file(video_filename):
    """Validate that video file exists and can be opened."""
    print(f"🎬 Validating video file: {video_filename}")
    
    if not os.path.exists(video_filename):
        raise FileNotFoundError(f"❌ Video file not found: {video_filename}")
    
    # Test if video can be opened
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"❌ Cannot open video file: {video_filename}")
    
    # Get basic video info
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    print(f"✅ Video validated: {width}x{height}, {frame_count} frames, {fps:.2f} FPS")
    return {
        'width': width,
        'height': height, 
        'frame_count': frame_count,
        'fps': fps
    }

def process_segment_trajectory_data(data, start_frame, end_frame, segment_label):
    """Process and smooth trajectory data for a specific segment."""
    print(f"🔄 Processing segment {segment_label}: frames {start_frame} to {end_frame}")
    
    segment_frames = list(range(start_frame, end_frame + 1))
    segment_measurements = []
    valid_positions = []
    valid_frame_numbers = []
    
    # Extract measurements for each frame in segment
    for frame_num in segment_frames:
        frame_str = str(frame_num)
        
        if frame_str in data:
            x = data[frame_str]["x"] 
            y = data[frame_str]["y"]
            
            # Check for valid detections (not -1, -1)
            if x != -1 and y != -1:
                segment_measurements.append(np.array([float(x), float(y)]))
                valid_positions.append([float(x), float(y)])
                valid_frame_numbers.append(frame_num)
            else:
                segment_measurements.append(None)  # Invalid detection
        else:
            segment_measurements.append(None)  # Frame not in data
    
    valid_positions = np.array(valid_positions) if valid_positions else np.array([]).reshape(0, 2)
    
    print(f"   📊 Found {len(valid_positions)} valid detections out of {len(segment_frames)} frames")
    print(f"   📈 Coverage: {len(valid_positions)/len(segment_frames)*100:.1f}%")
    
    if len(valid_positions) < 2:
        print(f"   ⚠️  Not enough valid detections for segment {segment_label}")
        return None, None, None, None
    
    # Initialize and configure Kalman tracker
    tracker = SegmentKalmanTracker()
    
    # Set initial state to first valid position
    first_valid_idx = next(i for i, m in enumerate(segment_measurements) if m is not None)
    tracker.state[:2] = segment_measurements[first_valid_idx]
    
    # Estimate initial velocity from first two valid positions
    if len(valid_positions) >= 2:
        dt_frames = valid_frame_numbers[1] - valid_frame_numbers[0]
        if dt_frames > 0:
            vx_init = (valid_positions[1, 0] - valid_positions[0, 0]) / dt_frames
            vy_init = (valid_positions[1, 1] - valid_positions[0, 1]) / dt_frames
            tracker.state[2:] = [vx_init, vy_init]
    
    # Process all measurements through Kalman filter
    for measurement in segment_measurements:
        tracker.predict()
        tracker.update(measurement)
    
    smoothed_positions = np.array(tracker.history)
    confidence_scores = np.array(tracker.confidence)
    
    # Apply additional Savitzky-Golay smoothing for long trajectories
    if len(smoothed_positions) >= 9:
        window_length = min(9, len(smoothed_positions) // 2)
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:
            try:
                smoothed_positions[:, 0] = savgol_filter(smoothed_positions[:, 0], 
                                                       window_length=window_length, polyorder=2)
                smoothed_positions[:, 1] = savgol_filter(smoothed_positions[:, 1], 
                                                       window_length=window_length, polyorder=2)
                print(f"   ✨ Applied additional smoothing (window: {window_length})")
            except Exception as e:
                print(f"   ⚠️  Smoothing failed: {str(e)}")
    
    print(f"   ✅ Segment {segment_label} processed successfully")
    return segment_frames, smoothed_positions, confidence_scores, valid_positions

def create_game1_trajectory_overlay():
    """Create trajectory overlay specifically for game_1.mp4 with predefined segments."""
    
    # File configuration for game_1
    video_filename = "game_1.mp4"
    json_filename = "ball_markup_1.json" 
    output_filename = "game_1_with_trajectory_overlay.mp4"
    
    # Your predefined segments
    segments = {
        's1': (2212, 2273),
        's2': (2196, 2247),
        's3': (9228, 9308)
    }
    
    # Segment colors (BGR format for OpenCV)
    segment_colors = {
        's1': (0, 255, 0),      # Bright Green
        's2': (255, 100, 0),    # Bright Blue
        's3': (0, 165, 255),    # Orange
        's4': (255, 0, 255),    # Magenta
        's5': (0, 255, 255),    # Yellow
    }
    
    print("🏓 GAME_1.MP4 TRAJECTORY OVERLAY CREATOR")
    print("=" * 60)
    print(f"📹 Video: {video_filename}")
    print(f"📊 Data: {json_filename}")
    print(f"🎯 Output: {output_filename}")
    print(f"📋 Segments: {segments}")
    print("=" * 60)
    
    try:
        # Validate input files
        video_info = validate_video_file(video_filename)
        data = load_ball_markup_data(json_filename)
        
        # Process trajectory data for all segments
        print(f"\n🔄 PROCESSING TRAJECTORY DATA")
        print("-" * 40)
        
        segment_trajectories = {}
        
        for segment_label, (start_frame, end_frame) in segments.items():
            result = process_segment_trajectory_data(data, start_frame, end_frame, segment_label)
            
            if result[0] is not None:
                segment_frames, smoothed_positions, confidence_scores, valid_positions = result
                
                # Create frame-to-index mapping for fast lookup
                frame_to_idx = {frame: idx for idx, frame in enumerate(segment_frames)}
                
                segment_trajectories[segment_label] = {
                    'frames': segment_frames,
                    'positions': smoothed_positions,
                    'confidence': confidence_scores,
                    'valid_positions': valid_positions,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_to_idx': frame_to_idx,
                    'color': segment_colors.get(segment_label, (128, 128, 128))
                }
                
                print(f"   ✅ {segment_label.upper()}: {len(smoothed_positions)} trajectory points")
            else:
                print(f"   ❌ {segment_label.upper()}: No valid trajectory data")
        
        if not segment_trajectories:
            raise ValueError("❌ No valid trajectory data found for any segment!")
        
        print(f"\n✅ Successfully processed {len(segment_trajectories)} segments")
        
        # Create video overlay
        print(f"\n🎬 CREATING VIDEO OVERLAY")
        print("-" * 40)
        
        # Open input video
        cap = cv2.VideoCapture(video_filename)
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, video_info['fps'], 
                            (video_info['width'], video_info['height']))
        
        print(f"📝 Video writer initialized")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print(f"   Total frames to process: {video_info['frame_count']}")
        
        # Process video frame by frame
        video_frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Find active segments for current frame
            active_segments = []
            
            for segment_label, traj_data in segment_trajectories.items():
                if (video_frame_idx >= traj_data['start_frame'] and 
                    video_frame_idx <= traj_data['end_frame']):
                    
                    if video_frame_idx in traj_data['frame_to_idx']:
                        traj_idx = traj_data['frame_to_idx'][video_frame_idx]
                        pos = traj_data['positions'][traj_idx]
                        conf = traj_data['confidence'][traj_idx]
                        
                        active_segments.append({
                            'label': segment_label,
                            'position': pos,
                            'confidence': conf,
                            'traj_data': traj_data,
                            'traj_idx': traj_idx,
                            'color': traj_data['color']
                        })
            
            # Draw trajectory overlays for active segments
            for seg_info in active_segments:
                segment_label = seg_info['label']
                current_pos = seg_info['position']
                traj_data = seg_info['traj_data']
                traj_idx = seg_info['traj_idx']
                color = seg_info['color']
                confidence = seg_info['confidence']
                
                # Draw trajectory trail (adaptive length based on confidence)
                trail_length = min(40, traj_idx + 1)
                if confidence < 0.5:
                    trail_length = min(20, trail_length)
                
                start_idx = max(0, traj_idx - trail_length + 1)
                trail_positions = traj_data['positions'][start_idx:traj_idx + 1]
                
                # Draw trail with varying opacity
                for i in range(1, len(trail_positions)):
                    prev_pos = trail_positions[i-1]
                    curr_pos_trail = trail_positions[i]
                    
                    # Calculate line properties based on position in trail
                    age_factor = i / len(trail_positions)
                    thickness = max(1, int(4 * age_factor))
                    
                    # Draw trail line
                    cv2.line(frame, 
                            (int(prev_pos[0]), int(prev_pos[1])), 
                            (int(curr_pos_trail[0]), int(curr_pos_trail[1])), 
                            color, thickness)
                
                # Draw current ball position with confidence-based appearance
                ball_center = (int(current_pos[0]), int(current_pos[1]))
                
                if confidence > 0.7:
                    # High confidence: filled circle with ring
                    cv2.circle(frame, ball_center, 10, color, -1)  # Filled
                    cv2.circle(frame, ball_center, 14, color, 3)   # Ring
                elif confidence > 0.4:
                    # Medium confidence: ring only
                    cv2.circle(frame, ball_center, 8, color, 3)
                else:
                    # Low confidence: small ring
                    cv2.circle(frame, ball_center, 6, color, 2)
                
                # Add segment label
                label_pos = (ball_center[0] + 18, ball_center[1] - 12)
                cv2.putText(frame, segment_label.upper(), label_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add confidence indicator for debugging (optional)
                conf_text = f"{confidence:.2f}"
                conf_pos = (ball_center[0] + 18, ball_center[1] + 8)
                cv2.putText(frame, conf_text, conf_pos, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Add frame information overlay
            info_bg_color = (0, 0, 0)  # Black background
            info_text_color = (255, 255, 255)  # White text
            
            # Frame number
            cv2.rectangle(frame, (5, 5), (250, 35), info_bg_color, -1)
            cv2.putText(frame, f"Frame: {video_frame_idx}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_text_color, 2)
            
            # Active segments
            if active_segments:
                active_labels = [seg['label'].upper() for seg in active_segments]
                active_text = f"Active: {', '.join(active_labels)}"
                cv2.rectangle(frame, (5, 40), (300, 65), info_bg_color, -1)
                cv2.putText(frame, active_text, (10, 58), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write processed frame
            out.write(frame)
            processed_frames += 1
            
            # Progress reporting
            if video_frame_idx % 1000 == 0 or video_frame_idx < 10:
                progress = (video_frame_idx / video_info['frame_count']) * 100
                print(f"   📊 Progress: {progress:.1f}% (Frame {video_frame_idx:,})")
                
                if active_segments:
                    active_info = [f"{seg['label']}({seg['confidence']:.2f})" for seg in active_segments]
                    print(f"      🎯 Active: {', '.join(active_info)}")
            
            video_frame_idx += 1
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ VIDEO OVERLAY COMPLETE!")
        print("=" * 60)
        print(f"📁 Output saved as: {output_filename}")
        print(f"📊 Processed {processed_frames:,} frames")
        
        # Generate summary report
        print(f"\n📋 TRAJECTORY SUMMARY:")
        print("-" * 40)
        
        for segment_label, traj_data in segment_trajectories.items():
            frames_range = f"{traj_data['start_frame']}-{traj_data['end_frame']}"
            valid_count = len(traj_data['valid_positions'])
            total_count = len(traj_data['frames'])
            coverage = (valid_count / total_count) * 100
            avg_confidence = np.mean(traj_data['confidence'])
            
            print(f"{segment_label.upper()}: Frames {frames_range}")
            print(f"   Detections: {valid_count}/{total_count} ({coverage:.1f}%)")
            print(f"   Avg Confidence: {avg_confidence:.2f}")
            
            if len(traj_data['valid_positions']) >= 2:
                distances = np.sqrt(np.sum(np.diff(traj_data['valid_positions'], axis=0)**2, axis=1))
                total_distance = np.sum(distances)
                print(f"   Total Distance: {total_distance:.0f} pixels")
        
        return output_filename
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {str(e)}")
        print("💡 Make sure both game_1.mp4 and ball_markup_1.json are in the current directory")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_trajectory_preview_plot():
    """Create a preview plot of all trajectories before video processing."""
    print("📊 Creating trajectory preview plot...")
    
    try:
        # Load data
        data = load_ball_markup_data("ball_markup_1.json")
        
        # Segments
        segments = {
            's1': (2212, 2273),
            's2': (2196, 2247), 
            's3': (9228, 9308)
        }
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        plt.figure(figsize=(14, 10))
        
        # Main trajectory plot
        plt.subplot(2, 2, 1)
        
        for i, (segment_label, (start_frame, end_frame)) in enumerate(segments.items()):
            result = process_segment_trajectory_data(data, start_frame, end_frame, segment_label)
            
            if result[0] is not None:
                segment_frames, smoothed_positions, confidence_scores, valid_positions = result
                
                color = colors[i % len(colors)]
                
                # Plot smoothed trajectory
                plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], 
                        color=color, linewidth=3, alpha=0.8, 
                        label=f'{segment_label.upper()} (Frames {start_frame}-{end_frame})')
                
                # Plot raw detections
                if len(valid_positions) > 0:
                    plt.scatter(valid_positions[:, 0], valid_positions[:, 1], 
                               alpha=0.4, s=15, color=color)
                
                # Mark start and end points
                if len(smoothed_positions) > 0:
                    plt.scatter(smoothed_positions[0, 0], smoothed_positions[0, 1], 
                               color=color, s=120, marker='o', edgecolor='black', 
                               linewidth=2, label=f'{segment_label.upper()} Start')
                    plt.scatter(smoothed_positions[-1, 0], smoothed_positions[-1, 1], 
                               color=color, s=120, marker='s', edgecolor='black', 
                               linewidth=2, label=f'{segment_label.upper()} End')
        
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Ball Trajectory Segments - Overview')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Coverage analysis
        plt.subplot(2, 2, 2)
        segment_names = []
        coverage_percentages = []
        
        for segment_label, (start_frame, end_frame) in segments.items():
            result = process_segment_trajectory_data(data, start_frame, end_frame, segment_label)
            if result[0] is not None:
                segment_frames, smoothed_positions, confidence_scores, valid_positions = result
                coverage = (len(valid_positions) / len(segment_frames)) * 100
                segment_names.append(segment_label.upper())
                coverage_percentages.append(coverage)
        
        plt.bar(segment_names, coverage_percentages, color=['red', 'blue', 'green'][:len(segment_names)])
        plt.ylabel('Coverage (%)')
        plt.title('Detection Coverage by Segment')
        plt.ylim(0, 100)
        
        for i, v in enumerate(coverage_percentages):
            plt.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        preview_filename = "game_1_trajectory_preview.png"
        plt.savefig(preview_filename, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Preview plot saved as: {preview_filename}")
        return preview_filename
        
    except Exception as e:
        print(f"❌ Error creating preview plot: {str(e)}")
        return None

def main():
    """Main execution function for game_1.mp4 trajectory overlay."""
    
    print("🚀 STARTING GAME_1.MP4 TRAJECTORY OVERLAY CREATION")
    print("=" * 70)
    
    # Check if required files exist
    required_files = ["game_1.mp4", "ball_markup_1.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ MISSING REQUIRED FILES:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please ensure these files are in the current directory:")
        print(f"   Current directory: {os.getcwd()}")
        return None
    
    try:
        # Step 1: Create preview plot
        print("\n📊 STEP 1: Creating trajectory preview plot...")
        preview_file = create_trajectory_preview_plot()
        
        # Step 2: Create video overlay
        print("\n🎬 STEP 2: Creating video overlay...")
        output_file = create_game1_trajectory_overlay()
        
        if output_file:
            print(f"\n🎉 SUCCESS! TRAJECTORY OVERLAY COMPLETE!")
            print("=" * 70)
            print("📁 Generated files:")
            print(f"   🎬 Video: {output_file}")
            if preview_file:
                print(f"   📊 Preview: {preview_file}")
            
            print(f"\n🎯 Your trajectory overlay video is ready!")
            print(f"   Open '{output_file}' to see the ball trajectories overlaid on your video.")
            
            return output_file
        else:
            print("❌ Failed to create trajectory overlay")
            return None
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the complete trajectory overlay creation for game_1.mp4
    result = main()
    
    if result:
        print(f"\n✨ All done! Your trajectory overlay video is ready.")
        print(f"🎬 File: {result}")
    else:
        print(f"\n💥 Something went wrong. Check the error messages above.")
        print(f"💡 Make sure 'game_1.mp4' and 'ball_markup_1.json' are in: {os.getcwd()}")