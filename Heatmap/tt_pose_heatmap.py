"""
tt_pose_hip_heatmap.py

Requirements:
 - ultralytics (YOLO11): pip install ultralytics
 - opencv-python, numpy, matplotlib
 - (optional) scipy for gaussian_filter: pip install scipy

This script:
 - runs YOLO11n-pose in tracking mode over a video,
 - extracts hip keypoints per tracked person (left_hip idx=11, right_hip idx=12 in COCO order),
 - accumulates hip positions per identity,
 - writes per-player heatmap PNGs and a combined overlay on a sample frame.
"""

import argparse
import torch
import os
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def main(args):
    ensure_dir(args.out_dir)
    # Load the YOLO11n pose model
    # model file name for official pose variant: "yolo11n-pose.pt"
    model_name = args.model if args.model else "yolo11n-pose.pt"
    model = YOLO(model_name)

    # Run tracking on the video (tracker default shown in docs)
    # model.track will perform detection+association; returns a results iterator.
    print(f"Running model.track on {args.video} with model {model_name} ...")
    track_results = model.track(source=args.video,
                                tracker=args.tracker or "bytetrack",
                                conf=args.confidence,
                                stream=True,
                                device=args.device or None,
                                persist=True)  # persist True to return results for all frames

    # data structure: dict mapping track_id -> list of hip positions (x,y)
    player_hips = {}
    # also save a sample frame (first frame) for overlay reference
    sample_frame = None
    sample_frame_shape = None
    frame_index = 0

    for r in track_results:
        # Each 'r' is a Result object for one frame
        # Obtain the frame as numpy (r.orig_img or r.orig_frame)
        if frame_index < args.start_frame:
            frame_index += 1
            continue

        if args.end_frame > 0 and frame_index > args.end_frame:
            break

        frame = None
        if hasattr(r, "orig_img"):
            frame = r.orig_img
        elif hasattr(r, "orig_frame"):
            frame = r.orig_frame
        else:
            # try to fallback
            frame = None

        if frame is not None:
            if sample_frame is None:
                sample_frame = frame.copy()
            h, w = frame.shape[:2]
            sample_frame_shape = (h, w)

        # Attempt to extract tracked boxes/ids and keypoints robustly:
        # r.boxes (if present) contains BoxList with .id and .xyxy etc.
        # r.keypoints may contain keypoints per instance.
        ids = []
        kps_list = []
        # detected instances count:
        # Attempt 1: r.boxes and r.keypoints
        if hasattr(r, "boxes") and r.boxes is not None:
            try:
                # r.boxes.id may be available
                ids = getattr(r.boxes, "id", None)
                # r.keypoints.data often present as Nx(K*3)
                if hasattr(r, "keypoints") and r.keypoints is not None:
                    kps = getattr(r.keypoints, "data", None)
                    if kps is None:
                        # maybe r.keypoints itself is array-like
                        kps = r.keypoints
                    kps_list = kps
                else:
                    # No keypoints field: maybe keypoints are inside boxes or other attr
                    kps_list = []
            except Exception:
                ids = []
                kps_list = []

        # Fallback: r.masks or r.boxes.xyxy etc won't help for pose.
        # If we can't find ids/kps in that frame, skip
        if ids is None:
            # maybe tracker not returning ids; try r.boxes.xyxy and no ids: skip frame
            frame_index += 1
            continue

        # Cast to list
        # ids may be tensor-like; convert to python list
        parsed_ids = []
        try:
            parsed_ids = [int(x) for x in ids.tolist()] if hasattr(ids, "tolist") else [int(x) for x in ids]
        except Exception:
            # if ids is scalar or empty
            try:
                parsed_ids = [int(ids)]
            except Exception:
                parsed_ids = []

        # kps_list should have same length as parsed_ids. If not, attempt to read r.keypoints directly:
        if (kps_list is None) or (len(parsed_ids) != len(kps_list)):
            # try r.keypoints itself as list of arrays
            if hasattr(r, "keypoints") and r.keypoints is not None:
                try:
                    kps_list = list(r.keypoints)
                except Exception:
                    kps_list = []

        # Now iterate matched id,kp
        for idx, tid in enumerate(parsed_ids):
            if idx >= len(kps_list):
                continue
            kp = kps_list[idx]
            # Extract hip point
            if frame is not None:
                h, w = frame.shape[:2]
            else:
                h, w = (args.out_h or 720, args.out_w or 1280)
            hip = extract_hip_point_from_keypoints(kp, w, h)
            if hip is not None:
                if tid not in player_hips:
                    player_hips[tid] = []
                player_hips[tid].append((hip[0], hip[1]))
        frame_index += 1

    print(f"Processed frames, found {len(player_hips)} tracked identities with hip samples.")

    # Create density maps for each player
    if sample_frame is None:
        print("No frames were read from the video (check video path). Exiting.")
        return

    h, w = sample_frame.shape[:2]
    combined_map = np.zeros((h, w), dtype=np.float32)

    ensure_dir(os.path.join(args.out_dir, "per_player"))
    for tid, pts in player_hips.items():
        density = np.zeros((h, w), dtype=np.float32)
        for (x, y) in pts:
            add_point_to_map(density, x, y, radius=args.point_radius, intensity=1.0)
        # smooth the density
        if _HAS_SCIPY:
            density = gaussian_filter(density, sigma=args.sigma)
        else:
            # fall back to OpenCV blur
            kr = int(max(3, args.sigma * 2 + 1))
            if kr % 2 == 0: kr += 1
            density = cv2.GaussianBlur(density, (kr, kr), sigmaX=args.sigma)
        # normalize for visualization
        maxv = density.max() if density.max() > 0 else 1.0
        vis = (density / maxv * 255).astype(np.uint8)
        # save grayscale heatmap
        out_file = os.path.join(args.out_dir, "per_player", f"player_{tid:02d}_heatmap.png")
        cv2.imwrite(out_file, vis)
        print("Saved", out_file)
        combined_map += density

    # Save combined heatmap overlay on sample frame
    if combined_map.max() > 0:
        cmax = combined_map.max()
        norm = (combined_map / cmax * 255).astype(np.uint8)
    else:
        norm = (combined_map * 0).astype(np.uint8)
    # colorize (use matplotlib colormap)
    cmap = plt.get_cmap("jet")
    colored = cmap(norm / 255.0)[:, :, :3]  # RGB float [0..1]
    colored = (colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(sample_frame, 0.6, cv2.cvtColor(colored, cv2.COLOR_RGB2BGR), 0.4, 0)
    cv2.imwrite(os.path.join(args.out_dir, "combined_heatmap_overlay.png"), overlay)
    cv2.imwrite(os.path.join(args.out_dir, "combined_heatmap_gray.png"), norm)
    print("Saved combined heatmap images in", args.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to input video (table tennis match)")
    p.add_argument("--out_dir", default="outputs", help="Output directory")
    p.add_argument("--model", default=None, help="YOLO model path or name (default: yolo11n-pose.pt)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker (bytetrack, strongsort etc.)")
    p.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    p.add_argument("--device", default=None, help="Device id (e.g. 0) or 'cpu' to force CPU")
    p.add_argument("--point_radius", type=int, default=6, help="Radius used when adding hip points to density map")
    p.add_argument("--sigma", type=float, default=8.0, help="Gaussian sigma for smoothing density map")
    p.add_argument("--start_frame", type=int, default=0, help="Frame index to start processing from")
    p.add_argument("--end_frame", type=int, default=-1, help="Frame index to stop processing at (-1 means till end)")

    args = p.parse_args()
    main(args)
