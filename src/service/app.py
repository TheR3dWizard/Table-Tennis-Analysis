import io
import json
from typing import Iterator

import cv2
import numpy as np
from flask import Flask, jsonify, request, Response

from .config import load_from_env
from .predictor import BallPredictor


def create_app() -> Flask:
    cfg = load_from_env()
    predictor = BallPredictor(
        weights_path=cfg.weights_path,
        device=cfg.device,
        score_threshold=cfg.score_threshold,
        step=cfg.step,
        input_size=cfg.input_size,
        use_tracker=cfg.use_tracker,
        max_disp=cfg.max_disp,
    )

    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.post("/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "missing image"}), 400
        file = request.files["image"]
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "invalid image"}), 400
        pred = predictor.predict_frame(img)
        return jsonify(pred)

    @app.post("/predict-video")
    def predict_video():
        # Accept multipart file or JSON path
        video_path = None
        if "video" in request.files:
            # Save to a temp file
            file = request.files["video"]
            import tempfile, os
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            with os.fdopen(fd, "wb") as f:
                f.write(file.read())
            video_path = tmp_path
        else:
            data = request.get_json(silent=True) or {}
            video_path = data.get("path")
        if not video_path:
            return jsonify({"error": "missing video"}), 400

        def stream() -> Iterator[bytes]:
            try:
                for item in predictor.predict_video(video_path):
                    yield (json.dumps(item) + "\n").encode("utf-8")
            finally:
                pass

        return Response(stream(), mimetype="application/x-ndjson")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000)


