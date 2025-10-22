import numpy as np
from sklearn.cluster import KMeans

def analyze_bounce_frequency_and_timing(bounce_points, segment_frames, fps=30):
    """
    Analyze bounce frequency and timing.
    Args:
        bounce_points (list): List of tuples (frame_index, y_position, edge).
        segment_frames (np.ndarray): Array of frame numbers for the segment.
        fps (float): Frames per second of the video.
    Returns:
        dict: Bounce frequency, intervals (in frames and seconds), and timing stats.
    """
    if not bounce_points or len(bounce_points) < 2:
        return {
            'bounce_count': len(bounce_points),
            'intervals_frames': [],
            'intervals_seconds': [],
            'mean_interval_frames': None,
            'mean_interval_seconds': None
        }
    bounce_indices = [bp[0] for bp in bounce_points]
    bounce_frames = segment_frames[bounce_indices]
    intervals_frames = np.diff(bounce_frames)
    intervals_seconds = intervals_frames / fps
    mean_interval_frames = np.mean(intervals_frames)
    mean_interval_seconds = np.mean(intervals_seconds)
    return {
        'bounce_count': len(bounce_points),
        'intervals_frames': intervals_frames.tolist(),
        'intervals_seconds': intervals_seconds.tolist(),
        'mean_interval_frames': mean_interval_frames,
        'mean_interval_seconds': mean_interval_seconds
    }

def cluster_bounce_locations(bounce_points, smoothed_positions, n_clusters=2):
    """
    Cluster bounce locations using KMeans.
    Args:
        bounce_points (list): List of tuples (frame_index, y_position, edge).
        smoothed_positions (np.ndarray): Array of shape (n, 2) with [x, y] coordinates.
        n_clusters (int): Number of clusters.
    Returns:
        dict: Cluster centers and labels for each bounce.
    """
    if not bounce_points:
        return {'centers': [], 'labels': []}
    bounce_coords = np.array([smoothed_positions[bp[0]] for bp in bounce_points])
    kmeans = KMeans(n_clusters=min(n_clusters, len(bounce_coords)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(bounce_coords)
    centers = kmeans.cluster_centers_
    return {'centers': centers, 'labels': labels.tolist()}

def estimate_bounce_heights(smoothed_positions, bounce_points, window=5):
    """
    Estimate bounce heights by finding the local maximum y before and after each bounce.
    Args:
        smoothed_positions (np.ndarray): Array of shape (n, 2) with [x, y] coordinates.
        bounce_points (list): List of tuples (frame_index, y_position, edge).
        window (int): Number of frames before and after bounce to search for peak.
    Returns:
        list: List of estimated heights for each bounce.
    """
    heights = []
    for bp in bounce_points:
        idx = bp[0]
        start = max(0, idx - window)
        end = min(len(smoothed_positions), idx + window + 1)
        y_vals = smoothed_positions[start:end, 1]
        peak_y = np.max(y_vals)
        bounce_y = smoothed_positions[idx, 1]
        height = abs(peak_y - bounce_y)
        heights.append(height)
    return heights

def analyze_bounce_dynamics(smoothed_positions, bounce_points, segment_frames, fps=30, n_clusters=2, window=5):
    """
    Comprehensive bounce dynamics analysis.
    Returns:
        dict: Contains frequency/timing, location clustering, and height estimation.
    """
    freq_timing = analyze_bounce_frequency_and_timing(bounce_points, segment_frames, fps)
    clustering = cluster_bounce_locations(bounce_points, smoothed_positions, n_clusters)
    heights = estimate_bounce_heights(smoothed_positions, bounce_points, window)
    return {
        'frequency_timing': freq_timing,
        'location_clustering': clustering,
        'bounce_heights': heights
    }