# 对比多个 TensorRT engine 在同一输入下的输出数值一致性（量化误差评估）
# 必须在 trt 环境运行: conda activate trt
# 用法: python compare_engines.py <ref.engine> <other.engine> [<more.engine> ...]

import sys

import numpy as np
import tensorrt as trt
import torch


def load(path):
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(path, "rb") as f:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    return engine, ctx


def run(path, x):
    engine, ctx = load(path)
    name_in = engine.get_tensor_name(0)
    ctx.set_input_shape(name_in, tuple(x.shape))
    out_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                 if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
    name_out = out_names[0]
    out_shape = tuple(ctx.get_tensor_shape(name_out))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(name_in, x.data_ptr())
        ctx.set_tensor_address(name_out, y.data_ptr())
        ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return y.float().cpu().numpy()


def main():
    ref_path, others = sys.argv[1], sys.argv[2:]
    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640, device="cuda", dtype=torch.float32)

    ref = run(ref_path, x)
    print(f"参考 {ref_path.split('/')[-1]}")
    print(f"   输出 shape {ref.shape}")
    print(f"   框通道 [0:4]   min={ref[:, :4].min():.4f}  max={ref[:, :4].max():.4f}")
    print(f"   类别通道[4:]   min={ref[:, 4:].min():.4f}  max={ref[:, 4:].max():.4f}")
    print()

    for p in others:
        y = run(p, x)
        d = np.abs(y - ref)
        print(f"{p.split('/')[-1]}")
        # 分开统计：框回归通道值域大，类别分数通道值域 0~1，混在一起看会误导
        for label, sl in (("框通道 [0:4]", slice(0, 4)), ("类别通道 [4:]", slice(4, None))):
            dr = d[:, sl]
            print(f"   {label:14s} max|Δ|={dr.max():.6e}  mean|Δ|={dr.mean():.6e}")
        print(f"   全通道相对误差 max|Δ|/max|ref| = {d.max() / np.abs(ref).max() * 100:.4f}%")
        print(f"   类别分数最大值 (smoke 模型未充分训练，仅作参考) = {y[:, 4:].max():.6f}")
        print()


if __name__ == "__main__":
    main()
