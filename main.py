import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def resolve_model_path(model_path: str | None) -> str:
	"""Return a valid model path, trying common defaults when not provided."""
	if model_path:
		path = Path(model_path)
		if not path.exists():
			raise FileNotFoundError(f"Model file not found: {path}")
		return str(path)

	candidates = [
		Path("best.pt"),
	]

	for path in candidates:
		if path.exists():
			return str(path)

	raise FileNotFoundError(
		"No model provided and no default best.pt found. "
		"Use --model to pass a weights file."
	)


def run_realtime_plate_detection(
	model_path: str,
	video_source: str | int = 0,
	conf: float = 0.25,
	iou: float = 0.35,
	imgsz: int = 640,
	show_fps: bool = True,
	save_output: bool = False,
	output_path: str = "realtime_detection_output.mp4",
    target_fps: int | None = None,
	config_dict: dict | None = None,
	stop_event=None,
) -> None:
	"""Run real-time plate detection from webcam or video file."""
	detector = YOLO(model_path)
	cap = cv2.VideoCapture(video_source)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video source: {video_source}")

	writer = None
	if save_output:
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = cap.get(cv2.CAP_PROP_FPS)
		if fps is None or fps <= 0:
			fps = 30
		if target_fps is not None and target_fps > 0:
			fps = target_fps
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

	window_name = "Real-Time License Plate Detection"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, 1000, 600)
	prev_t = cv2.getTickCount()

	try:
		while True:
			if stop_event is not None and stop_event.is_set():
				break

			current_conf = config_dict.get("conf", conf) if config_dict else conf
			current_iou = config_dict.get("iou", iou) if config_dict else iou
			current_imgsz = config_dict.get("imgsz", imgsz) if config_dict else imgsz
			current_show_fps = config_dict.get("show_fps", show_fps) if config_dict else show_fps

			ok, frame = cap.read()
			if not ok:
				break

			results = detector.predict(
				source=frame,
				conf=current_conf,
				iou=current_iou,
				imgsz=current_imgsz,
				verbose=False,
			)

			vis = frame.copy()
			if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
				boxes = results[0].boxes.xyxy.cpu().numpy()
				scores = results[0].boxes.conf.cpu().numpy()
				classes = results[0].boxes.cls.cpu().numpy().astype(int)

				for box, score, cls_id in zip(boxes, scores, classes):
					x1, y1, x2, y2 = map(int, box)
					cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cls_name = detector.names.get(cls_id, str(cls_id))
					label = f"{cls_name} {score:.2f}"
					cv2.putText(
						vis,
						label,
						(x1, max(20, y1 - 6)),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.6,
						(0, 255, 0),
						2,
						cv2.LINE_AA,
					)

			if current_show_fps:
				cur_t = cv2.getTickCount()
				dt = (cur_t - prev_t) / cv2.getTickFrequency()
				prev_t = cur_t
				fps_text = 1.0 / dt if dt > 0 else 0.0
				cv2.putText(
					vis,
					f"FPS: {fps_text:.1f}",
					(10, 28),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.8,
					(0, 200, 255),
					2,
					cv2.LINE_AA,
				)

			cv2.imshow(window_name, vis)
			if writer is not None:
				writer.write(vis)

			# Break if 'q' is pressed or the window is closed
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
				break
	finally:
		cap.release()
		if writer is not None:
			writer.release()
		cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Real-time license plate detection")
	parser.add_argument("--model", type=str, default=None, help="Path to YOLO weights (.pt)")
	parser.add_argument(
		"--source",
		type=str,
		default="0",
		help="Video source: webcam index like 0, or video file path",
	)
	parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
	parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
	parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
	parser.add_argument(
		"--hide-fps",
		action="store_true",
		help="Disable FPS overlay in the output window",
	)
	parser.add_argument(
		"--save-output",
		action="store_true",
		help="Save rendered detection video to --output",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="realtime_detection_output.mp4",
		help="Saved output file path when --save-output is used",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	model_path = resolve_model_path(args.model)

	source: str | int
	source = int(args.source) if args.source.isdigit() else args.source

	run_realtime_plate_detection(
		model_path=model_path,
		video_source=source,
		conf=args.conf,
		iou=args.iou,
		imgsz=args.imgsz,
		show_fps=not args.hide_fps,
		save_output=args.save_output,
		output_path=args.output,
	)


if __name__ == "__main__":
	main()
