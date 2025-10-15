import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ServiceConfig:
    weights_path: Optional[str]
    device: str
    score_threshold: float
    step: int
    input_size: Tuple[int, int]
    use_tracker: bool
    max_disp: float


def load_from_env() -> ServiceConfig:
    weights_path = os.getenv("BLURBALL_WEIGHTS", "src/blurball_best")
    device = os.getenv("DEVICE", "cuda" if os.getenv("CUDA", "1") == "1" else "cpu")
    score_threshold = float(os.getenv("SCORE_THRESHOLD", "0.7"))
    step = int(os.getenv("DET_STEP", "1"))
    inp_w = int(os.getenv("INP_W", "512"))
    inp_h = int(os.getenv("INP_H", "288"))
    use_tracker = os.getenv("USE_TRACKER", "1") != "0"
    max_disp = float(os.getenv("TRACK_MAX_DISP", "30.0"))

    return ServiceConfig(
        weights_path=weights_path,
        device=device,
        score_threshold=score_threshold,
        step=step,
        input_size=(inp_w, inp_h),
        use_tracker=use_tracker,
        max_disp=max_disp,
    )


