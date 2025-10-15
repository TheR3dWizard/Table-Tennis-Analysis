import os
import os.path as osp
import sys
from pathlib import Path
from collections import deque, defaultdict
from typing import Deque, Dict, Generator, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure 'src' is on sys.path so absolute imports like 'detectors', 'trackers' work
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from detectors.blurball_detector import BlurBallDetector
from trackers.online_blur import OnlineTrackerBlur
from utils.image import get_affine_transform


class _AttrDict(dict):
    """Dict that supports attribute access and recursively wraps nested dicts."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = self._wrap(v)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("__dict__",):
            return super().__setattr__(name, value)
        self[name] = self._wrap(value)

    @staticmethod
    def _wrap(value):
        if isinstance(value, dict):
            return _AttrDict(value)
        return value


def _build_min_cfg(
    weights_path: Optional[str],
    device: str = "cpu",
    score_threshold: float = 0.7,
    step: int = 1,
    inp_width: int = 512,
    inp_height: int = 288,
    frames_in: int = 3,
    frames_out: int = 3,
):
    # Locate checkpoint file if a directory is provided (common in this repo)
    model_path_resolved = weights_path
    if model_path_resolved is not None and osp.isdir(model_path_resolved):
        candidate = osp.join(model_path_resolved, "best_model.pth.tar")
        if osp.exists(candidate):
            model_path_resolved = candidate

    # Minimal BlurBall MODEL graph copied from configs/model/blurball.yaml
    model_extra = {
        "FINAL_CONV_KERNEL": 1,
        "PRETRAINED_LAYERS": ["*"],
        "STEM": {
            "INPLANES": 64,
            "STRIDES": [1, 1],
        },
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [1],
            "NUM_CHANNELS": [32],
            "FUSE_METHOD": "SUM",
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2, 2],
            "NUM_CHANNELS": [16, 32],
            "FUSE_METHOD": "SUM",
        },
        "STAGE3": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2, 2, 2],
            "NUM_CHANNELS": [16, 32, 64],
            "FUSE_METHOD": "SUM",
        },
        "STAGE4": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [2, 2, 2, 2],
            "NUM_CHANNELS": [16, 32, 64, 128],
            "FUSE_METHOD": "SUM",
        },
        "DECONV": {
            "NUM_DECONVS": 0,
            "KERNEL_SIZE": [],
            "NUM_BASIC_BLOCKS": 2,
        },
    }

    cfg_model = {
            "name": "blurball",
            "frames_in": frames_in,
            "frames_out": frames_out,
            "inp_width": inp_width,
            "inp_height": inp_height,
            "out_width": inp_width,
            "out_height": inp_height,
            "rgb_diff": False,
            "out_scales": [0],
            "MODEL": {
                "EXTRA": model_extra,
                "INIT_WEIGHTS": True,
            },
        }
    # Wrap model dict to support attribute access used by models/blurball.py
    cfg_model_wrapped = _AttrDict(cfg_model)

    cfg = {
        "model": cfg_model_wrapped,
        "detector": {
            "model_path": model_path_resolved,
            "step": step,
            "postprocessor": {
                "name": "blurball",
                "score_threshold": score_threshold,
                "scales": [0],
                "blob_det_method": "concomp",
                "use_hm_weight": True,
            },
        },
        "dataloader": {
            "heatmap": {
                # Matches default.yaml sigmas
                "sigmas": [2.5],
            }
        },
        "runner": {
            "device": device,
            "gpus": [0] if device == "cuda" else [],
        },
    }
    return cfg


class BallPredictor:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        score_threshold: float = 0.7,
        step: int = 1,
        input_size: Tuple[int, int] = (512, 288),
        use_tracker: bool = True,
        max_disp: float = 30.0,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        inp_w, inp_h = input_size
        self._cfg = _build_min_cfg(
            weights_path=weights_path,
            device=device,
            score_threshold=score_threshold,
            step=step,
            inp_width=inp_w,
            inp_height=inp_h,
            frames_in=3,
            frames_out=3,
        )

        self._detector = BlurBallDetector(self._cfg)
        self._frames_in = self._cfg["model"]["frames_in"]
        self._frames_out = self._cfg["model"]["frames_out"]
        self._input_wh = (self._cfg["model"]["inp_width"], self._cfg["model"]["inp_height"])
        self._out_scales = self._cfg["model"]["out_scales"]

        self._use_tracker = use_tracker
        self._tracker = OnlineTrackerBlur({"tracker": {"max_disp": max_disp}}) if use_tracker else None
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=self._frames_in)

    def _prepare_sample(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]

        # Maintain a rolling buffer; duplicate the edge frame until buffer is full
        if len(self._frame_buffer) == 0:
            for _ in range(self._frames_in):
                self._frame_buffer.append(frame_bgr)
        else:
            self._frame_buffer.append(frame_bgr)
            while len(self._frame_buffer) < self._frames_in:
                self._frame_buffer.append(frame_bgr)

        # Build affine transforms: inverse maps output -> original image
        input_w, input_h = self._input_wh
        out_w, out_h = self._cfg["model"]["out_width"], self._cfg["model"]["out_height"]
        c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        s = float(max(h, w))
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h], inv=0)
        trans_output_inv = {scale: get_affine_transform(c, s, 0, [out_w, out_h], inv=1) for scale in self._out_scales}

        # Warp frames and stack along channel dim (C=3*frames_in)
        imgs = []
        for img in list(self._frame_buffer):
            warped = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
            imgs.append(warped)
        imgs = np.stack(imgs, axis=0)  # (T, H, W, C)
        imgs = imgs.transpose(0, 3, 1, 2)  # (T, C, H, W)
        imgs = imgs.reshape(1, -1, input_h, input_w)  # (B=1, C=3*T, H, W)

        # Normalize like build_img_transforms (ImageNet stats)
        imgs = imgs.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        # Apply per 3-channel slice
        img_slices = []
        for t in range(self._frames_in):
            sl = imgs[:, 3 * t : 3 * (t + 1), :, :]
            sl = (sl - mean) / std
            img_slices.append(sl)
        imgs = np.concatenate(img_slices, axis=1)

        imgs_t = torch.from_numpy(imgs)
        affine_mats = {scale: torch.from_numpy(np.stack([trans_output_inv[scale]], axis=0)).float() for scale in self._out_scales}
        return imgs_t, affine_mats, (w, h)

    @torch.no_grad()
    def predict_frame(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        imgs_t, affine_mats, (w, h) = self._prepare_sample(frame_bgr)
        results, _ = self._detector.run_tensor(imgs_t, affine_mats)

        # Collect detections per emitted frame (eid)
        dets_by_eid = defaultdict(list)
        for eid in sorted(results[0].keys()):
            for det in results[0][eid]:
                dets_by_eid[eid].append(det)

        # If using tracker, feed sequentially; otherwise pick best score in last eid
        if self._use_tracker:
            if self._tracker is None:
                self._tracker = OnlineTrackerBlur({"tracker": {"max_disp": 30.0}})
            # We only return the last frame_out prediction (most recent)
            xy_obj = None
            for eid in sorted(dets_by_eid.keys()):
                xy_obj = self._tracker.update(dets_by_eid[eid])
            pred = xy_obj if xy_obj is not None else {"x": -1, "y": -1, "score": 0.0, "visi": False, "angle": 0.0, "length": 0.0}
        else:
            # Choose best by highest score in last eid
            last_eid = max(dets_by_eid.keys())
            best = None
            best_score = -1e9
            for det in dets_by_eid[last_eid]:
                if det["score"] > best_score:
                    best = det
                    best_score = det["score"]
            if best is None:
                pred = {"x": -1, "y": -1, "score": 0.0, "visi": False, "angle": 0.0, "length": 0.0}
            else:
                pred = {
                    "x": float(best["xy"][0]),
                    "y": float(best["xy"][1]),
                    "score": float(best.get("score", 0.0)),
                    "visi": True,
                    "angle": float(best.get("angle", 0.0)),
                    "length": float(best.get("length", 0.0)),
                }

        # Clamp to image bounds and cast ints for x,y
        x = int(max(0, min(pred["x"], w)))
        y = int(max(0, min(pred["y"], h)))
        return {
            "x": x,
            "y": y,
            "score": float(pred.get("score", 0.0)),
            "visi": bool(pred.get("visi", False)),
            "angle": float(pred.get("angle", 0.0)),
            "length": float(pred.get("length", 0.0)),
        }

    @torch.no_grad()
    def predict_video(self, video_path: str) -> Generator[Dict[str, float], None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        if self._use_tracker and self._tracker is not None:
            self._tracker.refresh()
        self._frame_buffer.clear()

        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pred = self.predict_frame(frame)
                pred_out = {
                    "frame": frame_index,
                    **pred,
                }
                frame_index += 1
                yield pred_out
        finally:
            cap.release()


