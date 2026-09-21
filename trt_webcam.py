# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Real-time webcam detection with a pure-TensorRT YOLO26 engine + live FPS overlay.

Pure TensorRT (no ultralytics runtime): deserialize engine -> bind GPU buffers ->
run async inference -> letterbox-aware decode -> NMS -> draw -> show.

Run inside the `trt` conda env (needs `import tensorrt`):
    conda activate trt

    # 默认：FP16 engine + 摄像头
    python trt_webcam.py

    # QAT 训出来的 INT8 engine + 摄像头
    python trt_webcam.py --engine runs/detect/runs/qat/smoke/weights/best-int8.engine

    # 没摄像头时用视频文件
    python trt_webcam.py --engine <engine> --source test_cam.mp4

    # 一次对比全部 engine（同一段视频，观察 FPS 差异）
    for e in yolo26n-my.engine yolo26n-fp16.engine \
             runs/detect/runs/qat/smoke/weights/best-int8.engine /tmp/int8-fp16cast.engine; do
        echo "=== $e ==="; python trt_webcam.py --engine "$e" --source test_cam.mp4
    done

Other flags: --conf 0.25  --imgsz 640  --no-show

WSL note: WSL2 does not expose a USB webcam by default. Either forward it with
`usbipd-win` (usbipd bind/attach) or use a video file / RTSP source instead.
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import tensorrt as trt
import torch

# ----------------------------- config -----------------------------------------
# 可用的 engine（本机实测 1x3x640x640，RTX 4060）：
#   yolo26n-my.engine                                    FP32      2.077 ms
#   yolo26n-fp16.engine                                  FP16      1.131 ms  ← 默认
#   runs/detect/runs/qat/smoke/weights/best-int8.engine  QAT INT8  1.315 ms
#   /tmp/int8-fp16cast.engine                            INT8+FP16 1.122 ms
#   yolo26-cgba-fp32.engine                              CGBA 结构实验
ENGINE = "yolo26n-fp16.engine"
SRC = 0  # 0 = 摄像头, 或 "video.mp4", 或 "rtsp://..."
CAM_WIDTH = 640  # webcam capture width (WSL/usbipd needs explicit size)
CAM_HEIGHT = 480  # webcam capture height
CAM_FOURCC = "MJPG"  # MJPG is required on WSL/usbipd; YUYV times out
IMGSZ = 640
CONF = 0.25  # confidence threshold
IOU = 0.45  # NMS IoU threshold
MAX_DET = 100  # max boxes drawn
SHOW = True  # set False for headless (no cv2.imshow)

# trtexec 裸转出来的 engine **不带 metadata**（既无 task 也无 names），
# 不填这个列表只会显示 cls0 / cls5；填了才显示 person / bus。
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
CLASSES = COCO_NAMES  # 设成 None 则退回显示 "cls{id}"


class TRTDetector:
    """Minimal TensorRT wrapper for a YOLO detection engine."""

    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("failed to deserialize engine (TensorRT version mismatch?)")
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.Stream()  # dedicated stream (avoids default-stream warning)

        self.in_name = self.out_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        self.in_shape = tuple(engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(engine.get_tensor_shape(self.out_name))

        # persistent device buffers (allocate once)
        self.d_in = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")
        self.d_out = torch.empty(self.out_shape, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(self.in_name, self.d_in.data_ptr())
        self.context.set_tensor_address(self.out_name, self.d_out.data_ptr())

    def infer(self, x: np.ndarray) -> np.ndarray:
        """X: (1,3,H,W) float32 numpy in [0,1]. Returns raw output (1,84,8400)."""
        self.d_in.copy_(torch.from_numpy(x), non_blocking=True)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.d_out.cpu().numpy()


def letterbox(img: np.ndarray, new_shape: int = 640, color: tuple = (114, 114, 114)):
    """Resize keeping aspect ratio and pad to square. Returns img, ratio, (dw, dh)."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw, dh = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[dh : dh + nh, dw : dw + nw] = img
    return canvas, r, (dw, dh)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain numpy NMS. boxes=(N,4) xyxy, scores=(N,). Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)[:MAX_DET]


def postprocess(out: np.ndarray, r: float, pad: tuple, orig_wh: tuple):
    """Decode (84,8400) -> (boxes_xyxy, scores, classes) mapped back to original image."""
    pred = out[0]  # (84, 8400)
    xywh = pred[:4].T  # cx,cy,w,h (letterboxed pixel space)
    scores_all = pred[4:].T  # (8400, nc)
    conf = scores_all.max(axis=1)
    cls = scores_all.argmax(axis=1)

    m = conf > CONF
    xywh, conf, cls = xywh[m], conf[m], cls[m]
    if len(conf) == 0:
        return np.zeros((0, 4)), conf, cls

    cx, cy, w, h = xywh.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # un-letterbox: remove padding then divide by ratio
    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r
    ow, oh = orig_wh
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ow)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, oh)

    keep = nms_numpy(boxes, conf, IOU)
    return boxes[keep], conf[keep], cls[keep]


def draw(frame: np.ndarray, boxes, conf, cls, fps: float):
    """Draw boxes and an FPS overlay."""
    for (x1, y1, x2, y2), c, k in zip(boxes, conf, cls):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = CLASSES[int(k)] if CLASSES else f"cls{int(k)}"
        cv2.putText(frame, f"{label} {c:.2f}", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # FPS (top-left), with a dark backdrop for readability
    txt = f"FPS: {fps:5.1f}"
    cv2.rectangle(frame, (5, 5), (135, 32), (0, 0, 0), -1)
    cv2.putText(frame, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def main() -> None:
    global ENGINE, SRC, IMGSZ, CONF, SHOW
    ap = argparse.ArgumentParser(description="纯 TensorRT YOLO26 摄像头实时检测")
    ap.add_argument("--engine", default=ENGINE, help="TensorRT .engine 路径")
    ap.add_argument("--source", default=str(SRC), help="0=摄像头，或视频文件 / RTSP URL")
    ap.add_argument("--conf", type=float, default=CONF, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, help="engine 输入边长（默认自动读 engine）")
    ap.add_argument("--no-show", action="store_true", help="不弹窗（无头模式）")
    a = ap.parse_args()
    ENGINE = a.engine
    SRC = int(a.source) if str(a.source).isdigit() else a.source
    CONF, SHOW = a.conf, not a.no_show

    det = TRTDetector(ENGINE)
    print(f"engine loaded: input={det.in_shape} output={det.out_shape}")
    # 以 engine 自身的输入尺寸为准，避免 IMGSZ 与 engine 不匹配导致精度崩坏
    IMGSZ = int(det.in_shape[2])
    print(f"IMGSZ = {IMGSZ} (取自 engine 输入)  conf={CONF}")

    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        raise SystemExit(
            f"cannot open source {SRC!r}. On WSL forward the webcam via usbipd, or set SRC to a video/RTSP."
        )
    if isinstance(SRC, int):
        # WSL/usbipd webcams only stream reliably as MJPG at an explicit resolution;
        # the default YUYV path stalls with `select() timeout`.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
        print(f"camera set: {CAM_WIDTH}x{CAM_HEIGHT} {CAM_FOURCC}")

    fps_window = deque(maxlen=30)  # rolling FPS over last 30 frames
    print("press 'q' in the window to quit")
    misses = 0
    n_frames = 0  # 统计用（无头模式也能拿到性能数据）
    n_boxes = 0
    t_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 30:  # tolerate transient WSL/usbipd stalls before giving up
                print("frame grab failed / stream ended")
                break
            time.sleep(0.01)
            continue
        misses = 0

        # preprocess: letterbox -> RGB -> CHW -> normalize
        lb, r, pad = letterbox(frame, IMGSZ)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        x = np.ascontiguousarray(x)

        out = det.infer(x)
        boxes, conf, cls = postprocess(out, r, pad, (frame.shape[1], frame.shape[0]))

        # fps = 1 / (time for full frame: capture+preprocess+infer+postprocess+draw)
        fps_window.append(time.perf_counter() - t0)
        fps = 1.0 / (sum(fps_window) / len(fps_window))

        vis = draw(frame, boxes, conf, cls, fps)
        n_frames += 1
        n_boxes += len(boxes)
        if SHOW:
            cv2.imshow("YOLO26 TensorRT", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW:
        cv2.destroyAllWindows()

    # 无头模式下的性能汇总（便于横向对比不同 engine）
    if n_frames:
        wall = time.perf_counter() - t_start
        print(f"\n--- {ENGINE} ---")
        print(f"frames {n_frames}  wall {wall:.2f}s  {wall / n_frames * 1000:.1f} ms/frame  {n_frames / wall:.1f} FPS")
        print(f"boxes drawn {n_boxes} (avg {n_boxes / n_frames:.2f}/frame)")


if __name__ == "__main__":
    main()
