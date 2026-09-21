# 纯 TensorRT engine 推理速度基准 —— 不需要摄像头，也不需要视频文件。
# 用 CUDA Event 精确计时，并把「H2D 拷贝 / 推理 / D2H 拷贝」分开统计，
# 用来回答：换 engine 精度到底能省多少时间？瓶颈到底在哪一段？
#
# 必须在 trt 环境运行（需要 import tensorrt）:
#     conda activate trt
#
# 用法:
#     python bench_engine.py                                  # 默认对比本机几个 engine
#     python bench_engine.py a.engine b.engine                # 指定 engine
#     python bench_engine.py --iters 500 --warmup 100         # 调整迭代次数
#     python bench_engine.py --csv out.csv                    # 导出 CSV

import argparse
import statistics as st
import time

import numpy as np
import tensorrt as trt
import torch

DEFAULT_ENGINES = [
    "yolo26n-my.engine",
    "yolo26n-fp16.engine",
    "runs/detect/runs/qat/smoke/weights/best-int8.engine",
    "/tmp/int8-fp16cast.engine",
]


class Bench:
    """一个 engine 的推理基准。

    注意：TensorRT 11 的 Python 绑定里 `execute_async_v3` 每次调用有 ~3.5ms 固定开销
    （远高于 GPU 实际计算时间），所以"直接调用"测出来的是 CPU 绑定开销而非 GPU 算力。
    用 CUDA Graph 捕获后 replay，能把这个开销降到 ~0.02ms，得到真实 GPU 推理耗时。
    """

    def __init__(self, path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"反序列化失败: {path}")
        self.path = path
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        self.in_name = self.out_name = None
        for i in range(engine.num_io_tensors):
            n = engine.get_tensor_name(i)
            if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n
            else:
                self.out_name = n
        self.in_shape = tuple(engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(engine.get_tensor_shape(self.out_name))
        self.mem_mib = engine.device_memory_size_v2 / 1024 / 1024

        # 固定缓冲区，避免每轮重新分配（CUDA Graph 捕获也要求捕获前完成分配）
        self.d_in = torch.empty(self.in_shape, dtype=torch.float32, device="cuda")
        self.d_out = torch.empty(self.out_shape, dtype=torch.float32, device="cuda")
        self.h_in = torch.rand(self.in_shape, dtype=torch.float32)
        self.ctx.set_tensor_address(self.in_name, self.d_in.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.d_out.data_ptr())

    def _time(self, fn, iters: int) -> float:
        """用 CUDA Event 给 iters 次调用整体计时，返回单次毫秒数。"""
        fn()  # 触发一次，确保 shape 解析完成
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(self.stream)
        for _ in range(iters):
            fn()
        end.record(self.stream)
        self.stream.synchronize()
        return start.elapsed_time(end) / iters

    def _cpu_time(self, fn, iters: int) -> float:
        """纯 CPU 侧耗时（调用返回即计时结束，不等 GPU）。"""
        fn()
        self.stream.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters * 1000
        self.stream.synchronize()
        return dt

    def _make_graph(self):
        """把一次推理捕获成 CUDA Graph，绕开 Python 绑定的逐次调用开销。

        返回一个 replay 闭包 —— 必须在捕获所用的同一条流上 replay，
        否则 CUDA Event 会记在空流上，测出假的超低耗时。
        """
        for _ in range(50):  # 预热，确保所有 kernel/内存都已就绪
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        def replay():
            with torch.cuda.stream(self.stream):
                g.replay()

        for _ in range(20):
            replay()
        self.stream.synchronize()
        return replay

    def run(self, warmup: int, iters: int, reps: int = 5) -> dict:
        def med(fn):
            return st.median([self._time(fn, iters) for _ in range(reps)])

        def t_infer():
            self.ctx.execute_async_v3(self.stream.cuda_stream)

        def t_h2d():
            self.d_in.copy_(self.h_in, non_blocking=True)

        def t_d2h():
            self.d_out.cpu()

        def t_full():
            self.d_in.copy_(self.h_in, non_blocking=True)
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            self.d_out.cpu()

        self._time(t_infer, warmup)
        out = {
            "h2d": med(t_h2d),
            "infer_direct": med(t_infer),
            "infer_cpu": self._cpu_time(t_infer, iters),
            "d2h": med(t_d2h),
            "full": med(t_full),
        }

        # CUDA Graph：得到不受 Python 绑定污染的真实 GPU 推理耗时
        try:
            replay = self._make_graph()
            out["infer_graph"] = med(replay)
            out["infer_graph_cpu"] = self._cpu_time(replay, iters)
        except Exception as ex:  # noqa: BLE001
            out["infer_graph"] = float("nan")
            out["graph_err"] = f"{type(ex).__name__}: {ex}"
        return out


def main():
    ap = argparse.ArgumentParser(description="纯 TensorRT engine 推理速度基准")
    ap.add_argument("engines", nargs="*", default=None, help="engine 路径（默认内置几个）")
    ap.add_argument("--iters", type=int, default=300, help="每次测量的迭代次数")
    ap.add_argument("--warmup", type=int, default=50, help="预热次数")
    ap.add_argument("--reps", type=int, default=5, help="重复测量次数，取中位数")
    ap.add_argument("--csv", default=None, help="把结果导出成 CSV")
    a = ap.parse_args()

    paths = a.engines or DEFAULT_ENGINES
    rows = []
    for p in paths:
        try:
            b = Bench(p)
        except FileNotFoundError:
            print(f"跳过 {p}（文件不存在）")
            continue
        r = b.run(a.warmup, a.iters, a.reps)
        rows.append((p, b.in_shape, b.mem_mib, r))
        g = r["infer_graph"]
        print(
            f"{p}\n"
            f"   input {b.in_shape}  显存 {b.mem_mib:.1f} MiB\n"
            f"   ⭐ GPU 推理 (CUDA Graph) {g:7.3f} ms   ({1000 / g:7.1f} FPS)\n"
            f"   直接调 execute_async_v3 {r['infer_direct']:7.3f} ms  "
            f"(其中 CPU 绑定开销 {r['infer_cpu']:6.3f} ms)\n"
            f"   +H2D {r['h2d']:6.3f} ms   +D2H {r['d2h']:6.3f} ms   "
            f"完整环路 {r['full']:7.3f} ms\n"
        )
        if "graph_err" in r:
            print(f"   ⚠️ CUDA Graph 不可用: {r['graph_err']}\n")

    if rows:
        base = next((r for r in rows if "my.engine" in r[0] or "fp32" in r[0].lower()), rows[0])
        b0 = base[3]["infer_graph"]
        print(f"{'engine':40s} {'GPU推理':>9s} {'直接调用':>9s} {'FPS':>8s} {'相对加速':>9s}")
        print("-" * 80)
        for p, _, _, r in rows:
            g = r["infer_graph"]
            print(f"{p.split('/')[-1]:40s} {g:8.3f}ms {r['infer_direct']:8.3f}ms "
                  f"{1000 / g:7.1f} {b0 / g:8.2f}x")
        print()
        print("说明: 'GPU推理' 用 CUDA Graph 测得，是真实 GPU 计算时间；")
        print("      '直接调用' 含 TensorRT 11 Python 绑定每次 ~3.5ms 的固定开销。")

    if a.csv:
        with open(a.csv, "w") as f:
            f.write("engine,in_shape,mem_mib,h2d_ms,d2h_ms,infer_graph_ms,infer_direct_ms,"
                    "infer_cpu_ms,full_ms,gpu_fps\n")
            for p, s, m, r in rows:
                f.write(f"{p},{'x'.join(map(str, s))},{m:.2f},{r['h2d']:.4f},{r['d2h']:.4f},"
                        f"{r['infer_graph']:.4f},{r['infer_direct']:.4f},{r['infer_cpu']:.4f},"
                        f"{r['full']:.4f},{1000 / r['infer_graph']:.1f}\n")
        print(f"\n已导出 {a.csv}")


if __name__ == "__main__":
    main()
