import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_rally_length(bounce_points, segment_frames, fps, segment_label="s1", save_plot=True):
    """
    Estimate rally length by counting valid bounces and calculating duration using FPS.

    Args:
        bounce_points (list): List of tuples (frame_index, y_position, edge) from detect_bounce_points.
        segment_frames (np.ndarray): Array of frame numbers for the segment.
        fps (float): Frames per second of the video.
        segment_label (str): Label for the segment (e.g., 's1').
        save_plot (bool): Whether to save a visualization of the rally analysis.

    Returns:
        dict: Dictionary containing rally analysis results:
            - rally_duration: Duration of the rally in seconds.
            - bounce_count: Number of valid bounces.
    """
    print(f"\n🏓 Analyzing Rally Length for Segment {segment_label}")
    print("=" * 60)

    results = {
        'rally_duration': 0.0,
        'bounce_count': len(bounce_points)
    }

    rally_duration = (segment_frames[-1] - segment_frames[0]) / fps if len(segment_frames) > 0 else 0
    results['rally_duration'] = rally_duration
    print(f"Rally duration: {rally_duration:.2f} seconds")
    print(f"Number of bounces: {results['bounce_count']}")

    if save_plot:
        print("Creating rally length visualization...")
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.bar(['Rally Duration'], [rally_duration], color='blue')
        plt.ylabel('Duration (seconds)')
        plt.title(f'Rally Duration - Segment {segment_label}')

        plt.subplot(1, 2, 2)
        plt.bar(['Bounce Count'], [results['bounce_count']], color='green')
        plt.ylabel('Number of Bounces')
        plt.title(f'Bounce Count - Segment {segment_label}')

        plt.tight_layout()
        os.makedirs("trajectory_segments", exist_ok=True)
        filename = f"trajectory_segments/rally_length_{segment_label}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Rally length plot saved as: {filename}")
        plt.close()
        print("Closed rally length plot")

    print(f"Rally lasted {rally_duration:.2f} seconds with {results['bounce_count']} bounces")
    return results

def classify_shot_types(bounce_points, smoothed_positions, segment_frames, fps, segment_label="s1", save_plot=True):
    """
    Classify shots (serve, smash, lob) based on speed, bounce count, and trajectory shape.

    Args:
        bounce_points (list): List of tuples (frame_index, y_position, edge) from detect_bounce_points.
        smoothed_positions (np.ndarray): Array of shape (n, 2) with smoothed [x, y] coordinates.
        segment_frames (np.ndarray): Array of frame numbers for the segment.
        fps (float): Frames per second of the video.
        segment_label (str): Label for the segment (e.g., 's1').
        save_plot (bool): Whether to save a visualization of shot classifications.

    Returns:
        dict: Dictionary containing shot classification results:
            - shot_classifications: List of tuples (frame_index, shot_type, speed, curvature).
    """
    print(f"\n🏓 Classifying Shot Types for Segment {segment_label}")
    print("=" * 60)

    results = {
        'shot_classifications': []
    }

    if len(smoothed_positions) < 2:
        print("Insufficient data for shot classification")
        return results

    vx = np.gradient(smoothed_positions[:, 0])
    vy = np.gradient(smoothed_positions[:, 1])
    speeds = np.sqrt(vx**2 + vy**2)
    max_speed = np.max(speeds)

    ax = np.gradient(vx)
    ay = np.gradient(vy)
    curvatures = np.sqrt(ax**2 + ay**2)

    def classify_shot(bounce_idx, speed, curvature, bounce_count):
        if bounce_count <= 1 and speed > 0.7 * max_speed:
            return "smash"
        elif curvature > np.percentile(curvatures, 80) and speed < 0.5 * max_speed:
            return "lob"
        elif bounce_idx == 0 and speed > 0.5 * max_speed:
            return "serve"
        return "regular"

    bounce_indices = [point[0] for point in bounce_points]
    bounce_count = len(bounce_points)
    for i, bounce_idx in enumerate(bounce_indices):
        speed = speeds[bounce_idx] if bounce_idx < len(speeds) else 0
        curvature = curvatures[bounce_idx] if bounce_idx < len(curvatures) else 0
        shot_type = classify_shot(i, speed, curvature, bounce_count)
        results['shot_classifications'].append((bounce_idx, shot_type, speed, curvature))
        print(f"Frame {bounce_idx}: Classified as {shot_type} (speed={speed:.2f} px/frame, curvature={curvature:.2f})")

    if save_plot and results['shot_classifications']:
        print("Creating shot classification visualization...")
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], 'red', linewidth=2, label='Trajectory')
        shot_colors = {'serve': 'blue', 'smash': 'red', 'lob': 'green', 'regular': 'gray'}
        for frame_idx, shot_type, _, _ in results['shot_classifications']:
            plt.scatter(smoothed_positions[frame_idx, 0], smoothed_positions[frame_idx, 1], 
                       c=shot_colors[shot_type], s=100, label=shot_type, alpha=0.7)
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title(f'Shot Classifications - Segment {segment_label}')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True, alpha=0.3)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        os.makedirs("trajectory_segments", exist_ok=True)
        filename = f"trajectory_segments/shot_classifications_{segment_label}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Shot classification plot saved as: {filename}")
        plt.close()
        print("Closed shot classification plot")

    print(f"Shot classification completed: {len(results['shot_classifications'])} shots classified")
    return results