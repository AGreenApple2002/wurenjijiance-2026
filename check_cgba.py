"""结构改动自检脚本：对比官方 yolo26 与自定义 yolo26-cgba（LBDown + CGBlockAttn）。

检查内容：
  1) 两个配置能否正常 parse_model 并前向（end2end Detect 输出形状）
  2) 参数量 / FLOPs 对比
  3) 自定义模块是否真的进了网络
  4) （有 GPU 时）单帧延迟与峰值显存对比

用法（在装了 ultralytics 依赖的环境里）：
    python check_cgba.py
"""

from __future__ import annotations

import time

import torch

from ultralytics import YOLO

BASE_CFG = "ultralytics/cfg/models/26/yolo26.yaml"
CUSTOM_CFG = "ultralytics/cfg/models/26/yolo26-cgba.yaml"
IMGSZ = 640
PAIRS = [("baseline(yolo26n)", BASE_CFG), ("custom(ldown+cgattn)", CUSTOM_CFG)]


def flops_of(model: torch.nn.Module, imgsz: int) -> float:
    """用 thop 统计 GFLOPs，未安装时返回 nan。"""
    try:
        from thop import profile
    except ImportError:
        return float("nan")
    x = torch.randn(1, 3, imgsz, imgsz)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    return macs * 2 / 1e9


def count_modules(model: torch.nn.Module) -> dict[str, int]:
    """统计关键模块的出现次数。"""
    out: dict[str, int] = {}
    for name in ("LBDown", "CGBlockAttn"):
        out[name] = sum(1 for m in model.modules() if type(m).__name__ == name)
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    sample_out = None
    for label, cfg in PAIRS:
        wrapper = YOLO(cfg)
        model = wrapper.model.eval()  # 先在 CPU 上统计参数量/FLOPs（thop 要求输入与模型同设备）
        params = sum(p.numel() for p in model.parameters())
        gflops = flops_of(model, IMGSZ)
        model = model.to(device)
        x = torch.randn(1, 3, IMGSZ, IMGSZ, device=device)
        with torch.no_grad():
            y = model(x)
        if isinstance(y, (list, tuple)):
            desc = " | ".join(str(tuple(t.shape)) for t in y if torch.is_tensor(t))
        elif torch.is_tensor(y):
            desc = str(tuple(y.shape))
        else:
            desc = str(type(y))
        if sample_out is None:
            sample_out = desc
        extra = count_modules(model)
        print(f"\n[{label}]")
        print(f"  params   : {params / 1e6:.3f} M")
        print(f"  gflops   : {gflops:.2f} (imgsz={IMGSZ})")
        print(f"  modules  : {extra}")
        print(f"  output   : {desc}")
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for _ in range(10):
                    model(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(50):
                    model(x)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 50 * 1000
            print(f"  latency  : {dt:.2f} ms/frame (batch=1, fp32, 50 iters)")
            print(f"  peak mem : {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print("\nOK: 两个配置都能正常前向。baseline 输出示例:", sample_out)


if __name__ == "__main__":
    main()
