'''
Perplexity generated code for table detection, 
uses YOLO. Very verbose. Only for video files.
'''

#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
	from ultralytics import YOLO
except Exception as import_error:  # pragma: no cover
	print(
		"Failed to import ultralytics. Install dependencies first: pip install ultralytics opencv-python",
		file=sys.stderr,
	)
	raise import_error


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Detect tables in a video stream using YOLO (Ultralytics).",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	# Removed --source; set video path inside main()
	parser.add_argument(
		"--model",
		type=str,
		default="yolov8n.pt",
		help="Path to YOLO model weights (.pt). Defaults to COCO-pretrained yolov8n.",
	)
	parser.add_argument(
		"--conf",
		type=float,
		default=0.5,
		help="Inference confidence threshold",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cpu",
		help="Device to run inference on (e.g., cpu, cuda:0)",
	)
	parser.add_argument(
		"--classes",
		type=str,
		default="",
		help=(
			"Comma-separated class names to consider as 'table'. If empty, will automatically "
			"match any class containing the substring 'table' (e.g., 'dining table')."
		),
	)
	parser.add_argument(
		"--view",
		action="store_true",
		help="Display the annotated video window",
	)
	parser.add_argument(
		"--save",
		action="store_true",
		help="Save annotated output to a video file",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="runs/table-detection/output.mp4",
		help="Output video path when --save is provided",
	)
	parser.add_argument(
		"--resize",
		type=int,
		default=0,
		help="Optional max width to resize frames for faster inference (0 keeps original)",
	)
	return parser.parse_args()


def open_video_capture(source: str) -> cv2.VideoCapture:
	if source.isdigit():
		cap = cv2.VideoCapture(int(source))
	else:
		cap = cv2.VideoCapture(source)
	return cap


def create_video_writer(
	output_path: Path,
	frame_width: int,
	frame_height: int,
	fps: float,
) -> Optional[cv2.VideoWriter]:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 24.0), (frame_width, frame_height))
	if not writer.isOpened():
		return None
	return writer


def find_table_class_ids(
	model_names: dict,
	override_class_names: Optional[List[str]] = None,
) -> List[int]:
	class_ids: List[int] = []
	for class_id, class_name in model_names.items():
		name_lower = str(class_name).lower()
		if override_class_names:
			if name_lower in override_class_names:
				class_ids.append(class_id)
		else:
			if "table" in name_lower:
				class_ids.append(class_id)
	return class_ids


def annotate_frame(
	frame: np.ndarray,
	boxes_xyxy: np.ndarray,
	classes: np.ndarray,
	confs: np.ndarray,
	class_names: dict,
	target_class_ids: List[int],
) -> np.ndarray:
	annotated = frame.copy()
	for idx in range(boxes_xyxy.shape[0]):
		class_id = int(classes[idx])
		if class_id not in target_class_ids:
			continue
		x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
		confidence = float(confs[idx])
		label = f"{class_names.get(class_id, class_id)} {confidence:.2f}"
		cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
		text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
		text_w, text_h = text_size
		cv2.rectangle(annotated, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), (0, 200, 0), -1)
		cv2.putText(
			annotated,
			label,
			(x1 + 3, y1 - 6),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(255, 255, 255),
			2,
			lineType=cv2.LINE_AA,
		)
	return annotated


def main() -> None:
	args = parse_arguments()

	# File input path set as a string here (not via CLI)
	# Update this path to your actual video file
	video_source = "assets/sample1.mp4"

	model_path = args.model
	confidence_threshold = float(args.conf)
	device = args.device

	model = YOLO(model_path)

	if args.classes:
		override = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
	else:
		override = None

	target_class_ids = find_table_class_ids(model.names, override)
	if not target_class_ids:
		print("Warning: No class ids matching 'table' found. Check --classes or model labels.", file=sys.stderr)

	cap = open_video_capture(video_source)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video source: {video_source}")

	original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0

	resize_width = int(args.resize) if args.resize and int(args.resize) > 0 else 0
	if resize_width > 0 and original_width > 0:
		scale = resize_width / float(original_width)
		resized_dims = (resize_width, int(original_height * scale))
	else:
		resized_dims = None

	writer = None
	if args.save:
		output_path = Path(args.output)
		writer = create_video_writer(output_path, original_width if not resized_dims else resized_dims[0], original_height if not resized_dims else resized_dims[1], input_fps)
		if writer is None:
			raise RuntimeError(f"Failed to open video writer at {output_path}")

	window_name = "Table Detection (YOLO)"
	if args.view:
		cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			if resized_dims is not None:
				frame_in = cv2.resize(frame, resized_dims, interpolation=cv2.INTER_LINEAR)
			else:
				frame_in = frame

			results = model.predict(
				frame_in,
				conf=confidence_threshold,
				device=device,
				verbose=False,
			)

			boxes = results[0].boxes
			if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
				annotated = annotate_frame(
					frame_in,
					boxes.xyxy.cpu().numpy(),
					boxes.cls.cpu().numpy(),
					boxes.conf.cpu().numpy(),
					model.names,
					target_class_ids,
				)
			else:
				annotated = frame_in

			if args.view:
				cv2.imshow(window_name, annotated)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break

			if writer is not None:
				writer.write(annotated)
	finally:
		cap.release()
		if writer is not None:
			writer.release()
		if args.view:
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
