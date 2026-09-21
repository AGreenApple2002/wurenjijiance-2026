# 反序列化 TensorRT engine，统计层的计算精度分布，确认量化是否真实生效
# 必须在 trt 环境运行（需要 import tensorrt）: conda activate trt
# 用法: python inspect_engine.py <model.engine> [<model2.engine> ...]

import sys
from collections import Counter

import tensorrt as trt


def inspect(path):
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print(f"{path}: ❌ 反序列化失败")
        return

    print(path.split("/")[-1])

    # TRT 10+ 用 I/O tensor API；层信息通过 num_layers/get_layer
    try:
        n_io = engine.num_io_tensors
        ios = [(engine.get_tensor_name(i), engine.get_tensor_mode(engine.get_tensor_name(i))) for i in range(n_io)]
        print(f"   I/O 张量 {n_io} 个: {[(n, str(m).split('.')[-1]) for n, m in ios]}")
    except AttributeError:
        pass

    # 层精度统计
    try:
        prec = Counter()
        for i in range(engine.num_layers):
            layer = engine.get_layer(i)
            prec[str(layer.precision).split(".")[-1]] += 1
        print(f"   层数 {engine.num_layers}")
        for k, v in prec.most_common():
            print(f"     {k:8s} {v:5d}")
    except Exception as e:
        print(f"   (层信息不可用: {type(e).__name__}: {e})")

    # 引擎设备显存需求（真实部署指标）
    try:
        print(f"   设备显存需求 {engine.device_memory_size / 1024 / 1024:.2f} MiB")
    except AttributeError:
        pass
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)
