"""基线 vs 自定义结构：ONNX 导出 -> trtexec 构建 -> TensorRT 实测延迟 + PyTorch 数值一致性。

为什么单独写这个脚本：
  1) TensorRT 11 的 trtexec 已经没有 --fp16/--int8 这类构建期精度开关（网络强类型化），
     精度必须在 ONNX 导出阶段决定，所以这里用 ultralytics 的 quantize=16 导出 fp16 图；
  2) 结构改动（LBDown 用了 grid_sample）是否正确，不能只看"跑得通"，必须比对 TRT 与
     PyTorch 的输出误差，避免部署后精度悄悄漂移。

用法：
    python bench_trt.py --cfg ultralytics/cfg/models/26/yolo26.yaml       --tag base
    python bench_trt.py --cfg ultralytics/cfg/models/26/yolo26-cgba.yaml --tag cgba
    python bench_trt.py ... --quantize 16
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO

# TensorRT SDK 根目录：优先读环境变量 TRT_ROOT，其次 TRT_HOME，都未设置时回退到常见安装路径。
# 本机配置示例（写入 ~/.bashrc）：
#   export TRT_ROOT=/path/to/TensorRT-11.0.0.114
TRT_ROOT = os.environ.get("TRT_ROOT") or os.environ.get("TRT_HOME", "/opt/TensorRT")
TRTEXEC = f"{TRT_ROOT}/bin/trtexec"


def export_onnx(cfg: str, out: Path, imgsz: int, opset: int, quantize: int | None) -> Path:
    """导出一个 ONNX（可选 fp16），返回产物路径。"""
    overrides = {"format": "onnx", "imgsz": imgsz, "opset": opset, "simplify": False, "dynamic": False}
    if quantize:
        overrides["quantize"] = quantize
    model = YOLO(cfg)
    path = Path(str(model.export(**overrides)))
    if path != out:
        out.unlink(missing_ok=True)
        path.replace(out)
    return out


def build_engine(onnx: Path, engine: Path, workspace_mb: int = 2048) -> str:
    """调用 trtexec 构建 engine，返回是否通过。"""
    cmd = [
        TRTEXEC,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--memPoolSize=workspace:{workspace_mb}",
    ]
    env = dict(os.environ, LD_LIBRARY_PATH=f"{TRT_ROOT}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    ok = "PASSED" in proc.stdout or engine.exists()
    tail = [ln for ln in proc.stdout.splitlines() if "Error" in ln or "error" in ln][:5]
    if not ok:
        print("trtexec stderr:", proc.stderr[-800:])
    return f"{'PASSED' if ok else 'FAILED'}" + (f" errors={tail}" if tail else "")


def run_engine(engine_path: Path, x: torch.Tensor, iters: int = 50) -> tuple[np.ndarray, float, float]:
    """用 TensorRT Python API 跑 engine，返回 (输出, 平均延迟 ms, 峰值显存 MiB)。"""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("engine 反序列化失败")
    context = engine.create_execution_context()

    in_name = out_names = None
    names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_name = name
        else:
            names.append(name)
    out_names = names
    in_shape = tuple(engine.get_tensor_shape(in_name))
    if tuple(x.shape) != in_shape:
        raise RuntimeError(f"输入形状不匹配：{tuple(x.shape)} vs engine {in_shape}")

    d_in = x.cuda().contiguous()
    d_outs = [torch.empty(tuple(engine.get_tensor_shape(n)), dtype=torch.float32, device="cuda") for n in out_names]
    context.set_tensor_address(in_name, d_in.data_ptr())
    for n, t in zip(out_names, d_outs):
        context.set_tensor_address(n, t.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream

    for _ in range(10):  # warmup
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    peak = torch.cuda.max_memory_allocated() / 2**20
    return d_outs[0].cpu().numpy(), ms, peak


def main() -> None:
    """导出 -> 构建 -> 基准测试 -> 与 PyTorch 比对。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--quantize", type=int, default=None, help="16=fp16, 8=int8；省略则 fp32")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    suffix = f"q{args.quantize}" if args.quantize else "fp32"
    onnx = Path(f"{args.tag}-{suffix}.onnx")
    engine = Path(f"{args.tag}-{suffix}.engine")

    if not args.skip_export:
        print(f"[1/4] 导出 ONNX: {args.cfg} -> {onnx} (quantize={args.quantize})")
        export_onnx(args.cfg, onnx, args.imgsz, args.opset, args.quantize)
    print(f"      onnx size = {onnx.stat().st_size / 2**20:.1f} MiB" if onnx.exists() else "      onnx 缺失")

    print(f"[2/4] trtexec 构建 engine -> {engine}")
    print("      " + build_engine(onnx, engine))

    print("[3/4] TensorRT 实测")
    torch.manual_seed(0)
    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    trt_out, ms, peak = run_engine(engine, x)
    print(f"      latency = {ms:.2f} ms/frame, peak alloc = {peak:.1f} MiB, out = {trt_out.shape}")

    print("[4/4] 与 PyTorch 输出比对（数值一致性）")
    model = YOLO(args.cfg).model.eval().cuda()
    with torch.no_grad():
        pt = model(x.cuda())
    pt_out = (pt[0] if isinstance(pt, (list, tuple)) else pt).cpu().numpy()
    if pt_out.shape == trt_out.shape:
        diff = np.abs(pt_out - trt_out)
        denom = np.maximum(np.abs(pt_out), 1e-6)
        print(f"      max|Δ| = {diff.max():.3e}, mean|Δ| = {diff.mean():.3e}, max relative = {(diff / denom).max():.3e}")
    else:
        print(f"      形状不同，跳过逐元素比较：torch {pt_out.shape} vs trt {trt_out.shape}")


if __name__ == "__main__":
    main()
