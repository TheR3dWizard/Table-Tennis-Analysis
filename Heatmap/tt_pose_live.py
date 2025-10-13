"""
tt_pose_live_heatmap.py
Visualizes YOLO11 pose hip positions as a live heatmap overlay on the video.

Usage:
  python tt_pose_live_heatmap.py --video input.mp4 --output live_heatmap.mp4
"""

import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from collections import defaultdict

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


LEFT_HIP, RIGHT_HIP = 11, 12  # COCO order


def extract_hip(kps, frame_w, frame_h):
    """Return avg hip position (x,y) or None."""
    import torch
    if isinstance(kps, torch.Tensor):
        kps = kps.detach().cpu().numpy()

    kps = np.array(kps).flatten().reshape(-1, 3)
    if kps[:, :2].max() <= 1.01:
        kps[:, 0] *= frame_w
        kps[:, 1] *= frame_h

    hips = []
    for idx in (LEFT_HIP, RIGHT_HIP):
        if idx < len(kps) and kps[idx, 2] > 0.2:
            hips.append(kps[idx, :2])
    if not hips:
        return None
    return np.mean(hips, axis=0)


def main(args):
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h}, {fps:.1f} FPS, {total_frames} frames")

    if args.output:
        out = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )
    else:
        out = None

    # Initialize heatmap arrays
    combined_map = np.zeros((h, w), dtype=np.float32)
    player_maps = defaultdict(lambda: np.zeros((h, w), dtype=np.float32))

    frame_index = 0
    for result in model.track(source=args.video, stream=True, tracker="bytetrack.yaml", persist=True, conf=args.conf):
        frame_index += 1
        frame = getattr(result, "orig_img", None)
        if frame is None:
            continue

        # Extract IDs and keypoints
        if not hasattr(result, "boxes") or result.boxes is None or result.keypoints is None:
            continue

        ids = getattr(result.boxes, "id", None)
        kps = getattr(result.keypoints, "data", None)
        if ids is None or kps is None:
            continue

        ids = ids.detach().cpu().numpy().astype(int)
        kps = kps.detach().cpu().numpy()

        for idx, track_id in enumerate(ids):
            hip = extract_hip(kps[idx], w, h)
            if hip is None:
                continue
            x, y = int(hip[0]), int(hip[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(player_maps[track_id], (x, y), 6, 1.0, -1)

        # Combine all players’ maps
        combined_map = np.zeros((h, w), dtype=np.float32)
        for pm in player_maps.values():
            combined_map += pm

        # Smooth map
        if _HAS_SCIPY:
            vis_map = gaussian_filter(combined_map, sigma=args.sigma)
        else:
            vis_map = cv2.GaussianBlur(combined_map, (0, 0), args.sigma)

        # Normalize + colorize
        if vis_map.max() > 0:
            norm = (vis_map / vis_map.max() * 255).astype(np.uint8)
        else:
            norm = vis_map.astype(np.uint8)
        color_map = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, color_map, 0.4, 0)

        # Show progress
        cv2.imshow("Live Heatmap", overlay)
        if out:
            out.write(overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    print("✅ Finished visualizing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output video file (optional)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--sigma", type=float, default=8.0, help="Gaussian blur sigma for smoothing")
    args = parser.parse_args()
    main(args)
