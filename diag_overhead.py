# 对照实验：这一环境的「Python -> CUDA」基础调用开销有多大？
# 用来判断 3.7ms 是 TensorRT 绑定特有，还是整个环境（WSL2）普遍如此。

import time

import tensorrt as trt
import torch

s = torch.cuda.Stream()
a = torch.zeros(1, device="cuda")
big = torch.zeros(1, 3, 640, 640, device="cuda")
N = 2000


def timeit(label, fn, n=N):
    for _ in range(200):
        fn()
    s.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n * 1000
    s.synchronize()
    print(f"{label:38s} {dt:8.4f} ms")


# --- 基线：纯 Python 空转 ---
timeit("① 纯 Python 空循环", lambda: None)


# --- ② 极小 torch kernel ---
def tiny():
    with torch.cuda.stream(s):
        a.add_(1)


timeit("② torch 极小 kernel (1 元素 add_)", tiny)


# --- ③ torch 中等 kernel ---
def mid():
    with torch.cuda.stream(s):
        big.add_(1)


timeit("③ torch 中等 kernel (1x3x640x640 add_)", mid)


# --- ④ H2D 拷贝 1x3x640x640 ---
h = torch.zeros(1, 3, 640, 640)


def h2d():
    with torch.cuda.stream(s):
        big.copy_(h, non_blocking=True)


timeit("④ H2D 拷贝 1x3x640x640", h2d)


# --- ⑤ TensorRT execute_async_v3 ---
rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
e = rt.deserialize_cuda_engine(open("yolo26n-fp16.engine", "rb").read())
ctx = e.create_execution_context()
n_in = e.get_tensor_name(0)
n_out = next(
    e.get_tensor_name(i)
    for i in range(e.num_io_tensors)
    if e.get_tensor_mode(e.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
)
d_in = torch.empty(tuple(e.get_tensor_shape(n_in)), dtype=torch.float32, device="cuda")
d_out = torch.empty(tuple(e.get_tensor_shape(n_out)), dtype=torch.float32, device="cuda")
ctx.set_tensor_address(n_in, d_in.data_ptr())
ctx.set_tensor_address(n_out, d_out.data_ptr())
timeit("⑤ TRT execute_async_v3", lambda: ctx.execute_async_v3(s.cuda_stream))

# --- ⑥ 重复 set_tensor_address 的开销 ---
timeit(
    "⑥ TRT set_tensor_address x2",
    lambda: (
        ctx.set_tensor_address(n_in, d_in.data_ptr()),
        ctx.set_tensor_address(n_out, d_out.data_ptr()),
    ),
)

print()
print("说明：① 是纯 Python 基线；②③ 是 torch 的 CUDA kernel 入队；")
print("      ⑤ 若远大于 ③，则开销来自 TensorRT Python 绑定本身。")
