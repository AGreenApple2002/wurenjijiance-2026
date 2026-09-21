# 给 QAT 导出的 INT8 ONNX 叠加 FP16 AutoCast：量化层跑 INT8，其余层跑 FP16 而非 FP32
# 目的：消除 INT8<->FP32 格式转换开销，让 INT8 engine 真正快于 FP16
# 必须在有 modelopt 的环境运行: conda activate base
# 用法: python qat_onnx_fp16cast.py <in.onnx> [out.onnx] [imgsz]

import sys

import onnx
import torch


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".onnx", ".fp16cast.onnx")
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640

    from modelopt.onnx import autocast

    m = onnx.load(src, load_external_data=False)
    input_name = m.graph.input[0].name

    print(f"AutoCast: {src} -> {dst} (low_precision_type=fp16, keep_io_types=True)")
    calib = {input_name: torch.randn(1, 3, imgsz, imgsz).cpu().numpy()}
    out = autocast.convert_to_mixed_precision(
        src,
        low_precision_type="fp16",
        keep_io_types=True,
        calibration_data=calib,
    )
    onnx.save(out, dst)
    print(f"✅ 已保存 {dst}")

    # 复核 Q/DQ 是否被保留
    from collections import Counter

    ops = Counter(n.op_type for n in out.graph.node)
    print(
        f"   节点 {len(out.graph.node)} | QuantizeLinear {ops.get('QuantizeLinear', 0)} | "
        f"DequantizeLinear {ops.get('DequantizeLinear', 0)}"
    )


if __name__ == "__main__":
    main()
