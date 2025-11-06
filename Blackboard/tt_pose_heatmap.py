"""
tt_pose_heatmap.py

Run Command:
python3 "Heatmap/tt_pose_heatmap.py" \
  --video "assets/rallies_02.mp4" \
  --model "yolo11n-pose.pt" \
  --out_dir "outputs/hip_heatmaps" \
  --num_players 2 --min_track_points 100

Requirements:
 - ultralytics (YOLO11): pip install ultralytics
 - opencv-python, numpy, matplotlib
 - (optional) scipy for gaussian_filter: pip install scipy

This script:
 - runs YOLO11n-pose in tracking mode over a video,
 - extracts hip keypoints per tracked person (left_hip idx=11, right_hip idx=12 in COCO order),
 - accumulates hip positions per identity,
 - writes per-player heatmap PNGs and a combined overlay on a sample frame.
 - analyze_video() function (demonstrated in demonstration.py) can be used as a standalone function to analyze a video and return the frame_map and overlay_path. 
"""

import argparse
import torch
import os
import hashlib
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from constants import Constants

try:
    from scipy.ndimage import gaussian_filter

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# COCO keypoint indices: left_hip=11, right_hip=12 (0-based COCO order)
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def add_point_to_map(density_map, x, y, radius=7, intensity=1.0):
    """Draw a filled circle on density_map (float) at integer coords (x,y)."""
    h, w = density_map.shape
    ix = int(round(x))
    iy = int(round(y))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return
    cv2.circle(density_map, (ix, iy), radius, float(intensity), thickness=-1)


def normalize_if_needed(kp, frame_w, frame_h):
    """
    ultralytics pose keypoints may be in absolute pixels or normalized [0..1].
    Heuristic: if all coords <=1 -> treat normalized and convert to pixels.
    kp shape expected (N,3) or (N*3) depending on API shape we get below.
    """
    arr = np.array(kp, dtype=float)
    if arr.size == 0:
        return arr
    # find max x or y
    maxv = arr.max()
    if maxv <= 1.01:
        # normalized -> convert
        arr = arr.reshape(-1, 3)
        arr[:, 0] *= frame_w
        arr[:, 1] *= frame_h
    else:
        arr = arr.reshape(-1, 3)
    return arr


def extract_hip_point_from_keypoints(kp_pts, frame_w, frame_h):
    """
    kp_pts expected as either:
      - numpy array shape (K,3) with x,y,confidence
      - 1D list [x1,y1,c1, x2,y2,c2, ...]
    Returns (x,y) or None if not found.
    """
    if kp_pts is None:
        return None
    if isinstance(kp_pts, torch.Tensor):
        kp_pts = kp_pts.detach().cpu().numpy()
    arr = np.array(kp_pts, dtype=float).flatten()
    if arr.size == 0:
        return None
    # ensure shape (K,3)
    if arr.size % 3 != 0:
        # unexpected format
        return None
    arr = arr.reshape(-1, 3)
    # If coordinates are normalized (0..1), scale:
    if arr[:, :2].max() <= 1.01:
        arr[:, 0] *= frame_w
        arr[:, 1] *= frame_h
    # pick hips
    lh = arr[LEFT_HIP_IDX] if LEFT_HIP_IDX < len(arr) else None
    rh = arr[RIGHT_HIP_IDX] if RIGHT_HIP_IDX < len(arr) else None
    chosen = None
    if lh is not None and lh[2] > 0.15:
        chosen = lh[:2]
    if rh is not None and rh[2] > 0.15:
        if chosen is None:
            chosen = rh[:2]
        else:
            # average
            chosen = (chosen + rh[:2]) / 2.0
    if chosen is None:
        return None
    return float(chosen[0]), float(chosen[1])


def analyze_video(
    video,
    start_frame=0,
    end_frame=-1,
    model=None,
    tracker="bytetrack.yaml",
    confidence=0.3,
    device=None,
    point_radius=6,
    sigma=8.0,
    out_dir="outputs",
    num_players=2,
    min_track_points=50,
):
    """
    Run tracking+pose on a video, keep the main players, build heatmaps, and
    return per-frame positions for the two selected players and the overlay path.

    Returns: (frame_map, overlay_path)
      - frame_map: {frame_index: {player1xposition:int, player1yposition:int,
                                  player2xposition:int, player2yposition:int}}
      - overlay_path: str full path to saved combined heatmap overlay PNG
    """
    ensure_dir(out_dir)
    model_name = model if model else Constants.YOLO11N_POSE_WEIGHTS_PATH
    yolo_model = YOLO(model_name)

    track_results = yolo_model.track(
        source=video,
        tracker=tracker or "bytetrack",
        conf=confidence,
        stream=True,
        device=device or None,
        persist=True,
    )

    player_hips = {}
    # Per-frame temporary store: frame_idx -> {track_id: (x,y)}
    per_frame_positions = {}
    sample_frame = None
    frame_index = 0

    for r in track_results:
        if frame_index < start_frame:
            frame_index += 1
            continue
        if end_frame > 0 and frame_index > end_frame:
            break

        frame = None
        if hasattr(r, "orig_img"):
            frame = r.orig_img
        elif hasattr(r, "orig_frame"):
            frame = r.orig_frame

        if frame is not None and sample_frame is None:
            sample_frame = frame.copy()

        ids = []
        kps_list = []
        if hasattr(r, "boxes") and r.boxes is not None:
            try:
                ids = getattr(r.boxes, "id", None)
                if hasattr(r, "keypoints") and r.keypoints is not None:
                    kps = getattr(r.keypoints, "data", None)
                    if kps is None:
                        kps = r.keypoints
                    kps_list = kps
                else:
                    kps_list = []
            except Exception:
                ids = []
                kps_list = []

        if ids is None:
            frame_index += 1
            continue

        parsed_ids = []
        try:
            parsed_ids = (
                [int(x) for x in ids.tolist()]
                if hasattr(ids, "tolist")
                else [int(x) for x in ids]
            )
        except Exception:
            try:
                parsed_ids = [int(ids)]
            except Exception:
                parsed_ids = []

        if (kps_list is None) or (len(parsed_ids) != len(kps_list)):
            if hasattr(r, "keypoints") and r.keypoints is not None:
                try:
                    kps_list = list(r.keypoints)
                except Exception:
                    kps_list = []

        frame_pos = {}
        for idx, tid in enumerate(parsed_ids):
            if idx >= len(kps_list):
                continue
            kp = kps_list[idx]
            if frame is not None:
                h, w = frame.shape[:2]
            else:
                h, w = (720, 1280)
            hip = extract_hip_point_from_keypoints(kp, w, h)
            if hip is not None:
                if tid not in player_hips:
                    player_hips[tid] = []
                player_hips[tid].append((hip[0], hip[1]))
                frame_pos[tid] = (hip[0], hip[1])

        if frame_pos:
            per_frame_positions[frame_index] = frame_pos
        frame_index += 1

    if sample_frame is None:
        return {}, ""

    # Choose main players
    counts = {tid: len(pts) for tid, pts in player_hips.items()}
    sorted_ids = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    filtered_ids = [tid for tid in sorted_ids if counts.get(tid, 0) >= min_track_points]
    if len(filtered_ids) < num_players:
        filtered_ids = sorted_ids[:num_players]
    else:
        filtered_ids = filtered_ids[:num_players]

    selected_ids = filtered_ids
    # Unique filename using hash(video path + start/end) + timestamp
    hasher = hashlib.md5()
    hasher.update(str(video).encode("utf-8"))
    hasher.update(str(start_frame).encode("utf-8"))
    hasher.update(str(end_frame).encode("utf-8"))
    hash_part = hasher.hexdigest()[:8]
    ts_part = datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_name = f"combined_heatmap_overlay_{hash_part}_{ts_part}.png"
    overlay_path = os.path.join(out_dir, overlay_name)

    # Build frame map for selected players (player1 -> selected_ids[0], player2 -> selected_ids[1] if exists)
    frame_map = {}
    pid1 = selected_ids[0] if len(selected_ids) > 0 else None
    pid2 = selected_ids[1] if len(selected_ids) > 1 else None
    for fidx in sorted(per_frame_positions.keys()):
        pos = per_frame_positions.get(fidx, {})
        x1, y1 = (-1, -1)
        x2, y2 = (-1, -1)
        if pid1 is not None and pid1 in pos:
            x1, y1 = int(round(pos[pid1][0])), int(round(pos[pid1][1]))
        if pid2 is not None and pid2 in pos:
            x2, y2 = int(round(pos[pid2][0])), int(round(pos[pid2][1]))
        frame_map[fidx] = {
            "player1x": x1,
            "player1y": y1,
            "player1z": -1,
            "player2x": x2,
            "player2y": y2,
            "player2z": -1,
            "combinedheatmappath": overlay_path,
        }

    # Heatmaps for selected players only
    h, w = sample_frame.shape[:2]
    combined_map = np.zeros((h, w), dtype=np.float32)
    ensure_dir(os.path.join(out_dir, "per_player"))
    for tid in selected_ids:
        pts = player_hips.get(tid, [])
        density = np.zeros((h, w), dtype=np.float32)
        for x, y in pts:
            add_point_to_map(density, x, y, radius=point_radius, intensity=1.0)
        if _HAS_SCIPY:
            density = gaussian_filter(density, sigma=sigma)
        else:
            kr = int(max(3, sigma * 2 + 1))
            if kr % 2 == 0:
                kr += 1
            density = cv2.GaussianBlur(density, (kr, kr), sigmaX=sigma)
        maxv = density.max() if density.max() > 0 else 1.0
        vis = (density / maxv * 255).astype(np.uint8)
        out_file = os.path.join(out_dir, "per_player", f"player_{tid:02d}_heatmap.png")
        cv2.imwrite(out_file, vis)
        combined_map += density

    if combined_map.max() > 0:
        cmax = combined_map.max()
        norm = (combined_map / cmax * 255).astype(np.uint8)
    else:
        norm = (combined_map * 0).astype(np.uint8)
    cmap = plt.get_cmap("jet")
    colored = cmap(norm / 255.0)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(
        sample_frame, 0.6, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR), 0.4, 0
    )


    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(os.path.join(out_dir, "combined_heatmap_gray.png"), norm)

    return frame_map, overlay_path


def main(args):
    frame_map, overlay_path = analyze_video(
        video=args.video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        model=args.model,
        tracker=args.tracker,
        confidence=args.confidence,
        device=args.device,
        point_radius=args.point_radius,
        sigma=args.sigma,
        out_dir=args.out_dir,
        num_players=args.num_players,
        min_track_points=args.min_track_points,
    )
    print("Saved combined heatmap overlay at", overlay_path)
    # Optionally dump JSON mapping for CLI usage
    # Users importing the function will get the dict returned directly


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--video", required=True, help="Path to input video (table tennis match)"
    )
    p.add_argument("--out_dir", default="outputs", help="Output directory")
    p.add_argument(
        "--model",
        default=None,
        help="YOLO model path or name (default: yolo11n-pose.pt)",
    )
    p.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker (bytetrack, strongsort etc.)",
    )
    p.add_argument(
        "--confidence", type=float, default=0.3, help="Detection confidence threshold"
    )
    p.add_argument(
        "--device", default=None, help="Device id (e.g. 0) or 'cpu' to force CPU"
    )
    p.add_argument(
        "--point_radius",
        type=int,
        default=6,
        help="Radius used when adding hip points to density map",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=8.0,
        help="Gaussian sigma for smoothing density map",
    )
    p.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Frame index to start processing from",
    )
    p.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Frame index to stop processing at (-1 means till end)",
    )
    p.add_argument(
        "--num_players",
        type=int,
        default=2,
        help="Number of primary players to keep based on track length",
    )
    p.add_argument(
        "--min_track_points",
        type=int,
        default=50,
        help="Minimum hip samples to consider a track a player",
    )

    args = p.parse_args()
    main(args)