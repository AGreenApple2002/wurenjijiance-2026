"""导出自定义结构 yolo26-cgba 到 ONNX，供 TensorRT(trtexec) 构建 engine。

用法：
    python export_cgba_onnx.py                 # 默认 opset 17, imgsz 640
    python export_cgba_onnx.py --opset 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    """按参数导出 ONNX 并打印产物路径。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="ultralytics/cfg/models/26/yolo26-cgba.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--simplify", action="store_true", help="需要安装 onnxslim")
    args = ap.parse_args()

    model = YOLO(args.cfg)
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
        half=False,
    )
    out = Path(str(path))
    print(f"\nONNX exported -> {out.resolve()}  exists={out.exists()} size={out.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
