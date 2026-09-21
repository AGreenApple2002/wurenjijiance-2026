# CGBA 改动导读（面向智能体 / 新会话）

> 本文件描述**本地非上游改动**。本仓库是 ultralytics 克隆，被改成无人机航拍小目标检测的
> 结构实验工程。任何智能体在新会话中接手前，先读本文件，再读 `EXPERIMENT_NOTES.md`。
>
> 最后更新：2026-09-14。所有数据均为本机实测，未做任何精度宣称（见第 7 节）。

## 1. 一句话摘要

在 YOLO26n 上新增两个结构模块 —— **LBDown（可学习双边下采样）** 与 **CGBlockAttn（粗粒度块注意力）**，
分别替换 backbone 前两级下采样、插入 P3/P4 级，并完成 ONNX 导出 → TensorRT 11 构建 → 实测延迟与
PyTorch 数值一致性比对的全链路验证。

## 2. 改动清单（含行号）

| 文件                                         | 状态            | 内容                                                                                                 |
| -------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------- |
| `ultralytics/nn/modules/cgba.py`             | **新增 203 行** | `LBDown`（40-113）、`CGBlockAttn`（116-203）；`__all__` 在第 36 行                                   |
| `ultralytics/cfg/models/26/yolo26-cgba.yaml` | **新增 56 行**  | 自定义模型配置（对照物：官方 `yolo26.yaml`）                                                         |
| `ultralytics/nn/modules/__init__.py`         | 修改            | L63 `from .cgba import CGBlockAttn, LBDown`；L120 `"CGBlockAttn"`；L168 `"LBDown"`                   |
| `ultralytics/nn/tasks.py`                    | 修改            | L44、L61 导入；**L2068-2069 加入 `parse_model` 的 `base_modules`**（关键，决定 c1/c2 与 width 缩放） |
| `check_cgba.py`                              | 新增            | 结构自检：前向、参数量、GFLOPs、延迟、峰值显存                                                       |
| `bench_trt.py`                               | 新增            | 导出 ONNX → trtexec 构建 → TRT 实测 → PyTorch 一致性比对                                             |
| `export_cgba_onnx.py`                        | 新增            | 仅导出 ONNX 的小工具                                                                                 |
| `EXPERIMENT_NOTES.md`                        | 新增            | 实验记录 + 真实困难清单                                                                              |

## 3. 模块语义

### 3.1 `LBDown(c1, c2, k=3, act=True)` — 可学习双边下采样

- **目的**：替换 stride-2 卷积，减少小目标响应在降采样时被平均掉。
- **数据流**：在 2x2 邻域取 4 个采样点 → 加权求和 → 1x1 卷积投影 → BN → SiLU。
- **权重**：`w_k = softplus(spatial_k) * exp(-(g_k - g_center)^2 / (2σ^2))`
  - `spatial`（4 个，`nn.Parameter`）：可学习空间先验；
  - `σ`（`log_sigma`，`nn.Parameter`）：强度相似度的高斯带宽；
  - `g = sigmoid(guide(x))`：1x1 卷积学到的强度响应（不是固定灰度差）；
  - 采样点位置 = 固定 2x2 网格 + `tanh(offset(x)) * 0.5`，`offset` 为 3x3 卷积预测的亚像素偏移。
- **实现要点**：用 `F.grid_sample`（`align_corners=True`、`padding_mode="border"`）做可微采样；
  偏移只在窗口中心处取（`off[:, :, 0::2, 0::2]`），避免对全分辨率偏移图二次采样。
- **导出影响**：ONNX 中产生 `GridSample` 节点（本配置共 16 个）。

### 3.2 `CGBlockAttn(c1, c2=None, block=8, num_heads=4, gate_bias=-2.0)` — 粗粒度块注意力

- **目的**：用极低算力获得跨区域长程上下文。
- **数据流**：`x` → `q/k/v` 1x1 卷积 → 按 `block x block` 不重叠块平均池化成粗粒度 token
  （token 数 ≈ H*W/block²）→ 多头自注意力（显式 `matmul + softmax`）→ 广播回像素 →
  逐像素门控 `sigmoid(gate(x))` → 残差 `x + y * gate`。
- **约束**：必须 `c1 == c2`（残差结构），否则 `__init__` 抛 `ValueError`。
- **为什么手写注意力**：`nn.MultiheadAttention` 的 ONNX 导出对 TensorRT 不友好。
- **初始近似恒等**：`gate.bias = -2.0`、`proj.bias = 0`，便于从预训练权重上继续训练。

## 4. yaml 接线与索引顺延（**改结构时最容易出错的地方**）

`ultralytics/cfg/models/26/yolo26-cgba.yaml` 相对官方 `yolo26.yaml`：

| 层                       | 官方                               | 本配置                                     |
| ------------------------ | ---------------------------------- | ------------------------------------------ |
| 0 / 1                    | `Conv [64,3,2]` / `Conv [128,3,2]` | **`LBDown [64]` / `LBDown [128]`**         |
| 5 / 8                    | —（无此层）                        | **`CGBlockAttn [512, 8, 4]`**（P3、P4 级） |
| head 各 `Concat` 的 from | 6 / 4 / 13 / 10                    | **8 / 5 / 15 / 12**                        |
| 检测头                   | `[[16,19,22], Detect]`             | **`[[18,21,24], Detect]`**                 |

新增层会让后面所有层的索引 +2（P3 处插一层）/+4（P4 处再插一层），**head 中每处 `from`
都必须同步顺延**，漏改会导致通道拼接数量错误。

## 5. 运行方式

```bash
cd ultralytics-main

# 环境：conda trt（Python 3.12，torch 2.6.0+cu124，tensorrt 11.0.0.114）
# TensorRT SDK：需自行下载并解压（trtexec 在该目录 bin/ 下），
#   本机通过 ~/.bashrc 设置：export TRT_ROOT=/path/to/TensorRT-11.0.0.114

conda run -n trt python check_cgba.py # 结构自检 + 参数量/GFLOPs/延迟/显存
conda run -n trt python bench_trt.py --cfg ultralytics/cfg/models/26/yolo26.yaml --tag base
conda run -n trt python bench_trt.py --cfg ultralytics/cfg/models/26/yolo26-cgba.yaml --tag cgba
# fp16 版本：追加 --quantize 16
```

`bench_trt.py` 依赖环境变量 `TRT_ROOT`（回退顺序：`TRT_ROOT` → `TRT_HOME` → `/opt/TensorRT`）。

## 6. 实测数据（RTX 4060 Laptop 8 GB，imgsz=640，batch=1）

| 配置                   | 参数量          | GFLOPs      | PyTorch         | TRT fp32        | TRT fp16        |
| ---------------------- | --------------- | ----------- | --------------- | --------------- | --------------- |
| baseline `yolo26n`     | 2.572 M         | 6.12        | 19.72 ms        | 2.05 ms         | 1.07 ms         |
| `+LBDown +CGBlockAttn` | 2.669 M (+3.8%) | 7.04 (+15%) | 27.45 ms (+39%) | 2.88 ms (1.41x) | 1.65 ms (1.55x) |

TRT 数值一致性（同输入，TRT vs PyTorch）：base fp32 `max|Δ|=3.05e-05`；cgba fp32 `1.83e-04`；
fp16 两者相对误差均约 `1.9e-03`。

**核心结论**：+15% GFLOPs 换来 +41% 延迟，瓶颈在 `grid_sample` 这类访存型算子，而非计算量。

## 7. 智能体必须遵守的边界

1. **不得报出任何精度/召回指标**：没有航拍数据集，未做训练与消融，**不存在 mAP 数字**。
2. 未验证：动态 shape、INT8、Jetson 等边缘设备、跨设备 engine 复用（engine 与 TRT 版本 + GPU 架构绑定）。
3. 这是一份**本地实验工程**，不是比赛作品代码；对外表述必须与实际来源一致。
4. 修改结构后必须重跑 `check_cgba.py` 与 `bench_trt.py`，并更新本文件第 6 节数据。

## 8. 已知踩坑（改代码前先看）

1. **TensorRT 11 没有 `--fp16` / `--int8` 构建期开关**（仅 `--noTF32`；`--stronglyTyped` 已废弃为 no-op），
   网络强类型化 → 精度必须由 ONNX 决定。
2. **ultralytics 新版导出参数**：`half=True` 已弃用，改用 `quantize=16`；`simplify` 需要 `onnxslim`。
3. **`thop.profile` 要求输入与模型同设备**：统计 FLOPs 要在 `.to("cuda")` 之前做，否则报
   `Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)`。
4. **trtexec 报 `Latency` 与 Python API 计时不可直接比较**：Python 侧逐次 `execute_async_v3`
   含调度开销（本机 cgba fp32 为 5.71 ms vs trtexec GPU 2.88 ms），比较请统一用 `trtexec --loadEngine`。
5. **`GridSample` 在 ONNX opset < 16 不可导出**；本工程用 opset 17。
6. **`bench_trt.py` 会用随机权重**（从 yaml 构建），只用于结构与性能验证，不代表训练效果。
