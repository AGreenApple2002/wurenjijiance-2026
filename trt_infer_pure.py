# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Minimal pure-TensorRT inference for a YOLO26 detection engine (no ultralytics runtime).

This script shows the *low-level* TensorRT workflow so you can see what actually happens
under the hood when you call `YOLO("xxx.engine").predict(...)`:

    1. deserialize the serialized engine
    2. create an execution context
    3. allocate GPU (input/output) buffers
    4. preprocess the image -> NCHW float32 -> copy to GPU
    5. run async inference
    6. copy output back and decode (xywh + class scores) + NMS

Run inside the `trt` conda env:
    conda activate trt
    python trt_infer_pure.py
"""

import cv2
import numpy as np
import tensorrt as trt
import torch

ENGINE = "yolo26n-my.engine"  # your TensorRT engine
IMGSZ = 640  # network input size
IMG = "ultralytics/assets/bus.jpg"  # test image
CONF = 0.25  # confidence threshold
IOU = 0.45  # NMS IoU threshold


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Letterbox + BGR->RGB + HWC->CHW + normalize + batch dim. Returns (1,3,H,W) float32."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # simple resize (ultralytics uses letterbox; resize is fine for a demo)
    img = cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]  # HWC -> 1CHW
    return np.ascontiguousarray(x)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain numpy NMS. boxes = (N,4) xyxy, scores = (N,). Returns kept indices."""
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
    return np.array(keep, dtype=np.int64)


def main() -> None:
    # --- 1. load engine ---------------------------------------------------------
    logger = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("failed to deserialize engine (version mismatch?)")
    context = engine.create_execution_context()
    print(f"engine loaded: {engine.num_io_tensors} I/O tensors")

    # discover input/output tensor names & shapes
    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_name = name
            print(f"  INPUT  {name} {shape}")
        else:
            out_name = name
            print(f"  OUTPUT {name} {shape}")
    in_shape = tuple(engine.get_tensor_shape(in_name))
    out_shape = tuple(engine.get_tensor_shape(out_name))

    # --- 2. prepare input -------------------------------------------------------
    img = cv2.imread(IMG)
    x = preprocess(img)  # (1,3,640,640)
    if x.shape != in_shape:
        raise RuntimeError(f"input shape {x.shape} != engine {in_shape}")

    # --- 3. allocate GPU buffers (torch tensors hold the device memory) --------
    d_in = torch.from_numpy(x).to("cuda").contiguous()
    d_out = torch.empty(out_shape, dtype=torch.float32, device="cuda")

    # --- 4. bind addresses & run ----------------------------------------------
    context.set_tensor_address(in_name, d_in.data_ptr())
    context.set_tensor_address(out_name, d_out.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    # warmup + run
    for _ in range(3):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        context.execute_async_v3(stream)
    end.record()
    torch.cuda.synchronize()
    print(f"TensorRT forward: {start.elapsed_time(end) / 20:.2f} ms/image")

    # --- 5. decode output = (1, 84, 8400): rows0-3 = cx,cy,w,h  rows4-83 = scores
    out = d_out.cpu().numpy()[0]  # (84, 8400)
    boxes_xywh = out[:4, :].T  # (8400, 4)
    class_scores = out[4:, :].T  # (8400, 80)
    conf = class_scores.max(axis=1)  # best class score per anchor
    cls = class_scores.argmax(axis=1)

    mask = conf > CONF
    boxes_xywh, conf, cls = boxes_xywh[mask], conf[mask], cls[mask]
    print(f"candidates after conf>{CONF}: {len(boxes_xywh)}")

    # xywh(center) -> xyxy
    cx, cy, w, h = boxes_xywh.T
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # --- 6. class-agnostic NMS (simple demo) -----------------------------------
    keep = nms_numpy(boxes_xyxy, conf, IOU)
    print(f"after NMS: {len(keep)} boxes")
    for k in keep:
        print(f"  class={cls[k]:>2d} conf={conf[k]:.2f} box={boxes_xyxy[k].round(1)}")

    # --- 7. draw ----------------------------------------------------------------
    vis = img.copy()
    for k in keep:
        x1, y1, x2, y2 = boxes_xyxy[k].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            vis, f"cls{cls[k]} {conf[k]:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1
        )
    out_path = "trt_pure_result.jpg"
    cv2.imwrite(out_path, vis)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
