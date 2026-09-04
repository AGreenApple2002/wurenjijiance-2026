from pathlib import Path
import argparse

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_WEIGHTS = (
    "/home/ps/ztz/aReasearch/repo/ultralytics-main/"
    "runs/detect/MyVisDrone/myvisdrone_exp12/weights/best.pt"
)


def video_side_by_side_predict(
    input_video: str,
    output_video: str,
    weights: str = DEFAULT_WEIGHTS,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str = "cpu",
) -> str:
    model = YOLO(weights)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"无法写出视频: {output_video}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            original = frame.copy()
            result = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )[0]

            pred_vis = result.plot()
            merged = np.hstack([original, pred_vis])
            writer.write(merged)
    finally:
        cap.release()
        writer.release()

    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MyVisDrone 视频检测并拼接原视频与预测视频。")
    parser.add_argument("input_video", help="输入视频路径，例如 input.mp4")
    parser.add_argument(
        "--output-video",
        default="outputs/myvisdrone_compare.mp4",
        help="输出视频路径，默认 outputs/myvisdrone_compare.mp4",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="模型权重路径",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", default=0, help='推理设备，CPU 用 "cpu"，GPU 用 "0"')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = video_side_by_side_predict(
        input_video=args.input_video,
        output_video=args.output_video,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
    print(f"输出视频: {out}")
