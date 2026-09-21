# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CGBA 结构（LBDown + CGBlockAttn）推理演示脚本。.

⚠️ 重要前提
    本脚本**不做训练**，所以：
      - 从 yaml 构建 = 随机权重 → 输出张量形状正确，但**类别分数≈0，检测不出任何目标**
      - 可选 --weights 载入预训练权重，但因为 CGBA 插入了新层、head 索引整体顺延，
        只有约 9% 的键能匹配（位置错位被丢弃）→ 仍然**没有可用的检测结果**

    因此本脚本的用途是**验证结构与部署链路**（形状/算子/导出/基准），
    而**不是**获得有意义的检测。要得到真实检测必须先在数据集上训练。

用法（conda 环境 trt）：
    python cgba_infer.py                                  # 随机权重，看输出形状与统计
    python cgba_infer.py --conf 0.0001 --save             # 强制出框（说明框是无意义的）
    python cgba_infer.py --weights yolo26n.pt             # 尝试部分迁移（约 9% 键匹配）
"""

from __future__ import annotations

import argparse

import torch

from ultralytics import YOLO

CFG = "ultralytics/cfg/models/26/yolo26-cgba.yaml"


def describe_structure(model) -> None:
    """打印 CGBA 模块数量与参数量，确认自定义层真的进了网络。."""
    names = {}
    for m in model.model.modules():
        n = type(m).__name__
        if n in {"LBDown", "CGBlockAttn"}:
            names[n] = names.get(n, 0) + 1
    params = sum(p.numel() for p in model.model.parameters())
    print(f"  自定义模块 : {names}")
    print(f"  参数量     : {params / 1e6:.3f} M")


def raw_forward_stats(model, imgsz: int = 640) -> None:
    """跑一次原始前向，打印 output0 的数值分布（判断模型是否"有话说"）。."""
    net = model.model.eval()
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        y = net(x)
    out = y[0] if isinstance(y, (list, tuple)) else y
    if not torch.is_tensor(out):
        print("  原始前向 : 非张量输出", type(out))
        return
    arr = out.cpu().numpy()
    if arr.ndim == 3:  # (B, 84, 8400) -> (84, 8400)
        arr = arr[0]
    boxes, scores = arr[:4], arr[4:]
    print(f"  原始前向 : shape={tuple(out.shape)}")
    print(f"  框坐标   : {boxes.min():.2f} ~ {boxes.max():.2f}（像素，无意义）")
    print(f"  类别分数 : {scores.min():.4f} ~ {scores.max():.4f}")
    print(f"  → 最大类别分数 {scores.max():.4f}；分数≈0 说明模型未训练，检测不出目标")


def try_load_weights(model, weights: str) -> None:
    """尝试部分迁移预训练权重，报告匹配率。."""
    sd = torch.load(weights, map_location="cpu", weights_only=False)["model"].state_dict()
    new = model.model.state_dict()
    matched = {k: v for k, v in sd.items() if k in new and v.shape == new[k].shape}
    model.model.load_state_dict(matched, strict=False)
    pct = len(matched) / len(sd) * 100
    print(f"  权重迁移 : {len(matched)}/{len(sd)} 键匹配（{pct:.1f}%）")
    if pct < 50:
        print("             ⚠️ 匹配率低：新层导致 head 索引顺延，大部分键位置错位被丢弃")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=CFG, help="模型 yaml")
    ap.add_argument("--weights", default=None, help="可选：预训练权重（部分加载）")
    ap.add_argument("--source", default="ultralytics/assets/bus.jpg", help="输入图片")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--save", action="store_true", help="保存可视化结果")
    args = ap.parse_args()

    print(f"[1] 从 yaml 构建模型: {args.cfg}")
    model = YOLO(args.cfg)
    describe_structure(model)

    if args.weights:
        print(f"[2] 载入预训练权重: {args.weights}")
        try_load_weights(model, args.weights)
    else:
        print("[2] 未提供 --weights → 使用随机权重")

    print("[3] 原始前向统计")
    raw_forward_stats(model, args.imgsz)

    print(f"[4] ultralytics 推理 (conf={args.conf})")
    res = model.predict(args.source, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
    print(f"  检测数   : {len(res.boxes)}")
    if len(res.boxes):
        conf = res.boxes.conf.cpu().numpy()
        print(f"  置信度   : {conf.min():.4f} ~ {conf.max():.4f}")
        print("             ⚠️ 未训练模型的框是随机噪声，不具备检测意义")
    print("  耗时     : " + ", ".join(f"{k}={v:.1f}ms" for k, v in res.speed.items()))
    if args.save:
        res.save(filename="cgba_infer_result.jpg")
        print("  已保存   : cgba_infer_result.jpg")

    print("\n提示：本脚本用于验证结构与部署链路；要得到真实检测必须先在数据集上训练。")


if __name__ == "__main__":
    main()
