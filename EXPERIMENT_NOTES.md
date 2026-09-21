# YOLO26 结构改动实验记录（LBDown + CGBlockAttn）

> 本地实验记录，**不是比赛作品代码**。用于记录"可学习双边下采样 + 粗粒度块注意力"两个结构改动
> 在 YOLO26n 上的实现、部署链路与实测代价。数据为 2026-09-14 在本机实测。

## 1. 环境

| 项目          | 值                                                 |
| ------------- | -------------------------------------------------- |
| GPU           | NVIDIA GeForce RTX 4060 Laptop (8 GB, Ada)         |
| 系统          | WSL2 + Linux                                       |
| 训练/导出环境 | conda `trt`：Python 3.12.14，torch 2.6.0+cu124     |
| TensorRT      | 11.0.0.114（完整 SDK，`trtexec` 命令行构建）       |
| ultralytics   | 8.4.143（本地克隆，项目根目录 `ultralytics-main`） |

## 2. 改动内容

新增 `ultralytics/nn/modules/cgba.py` 两个模块，并在 `ultralytics/cfg/models/26/yolo26-cgba.yaml` 中接线：

- **LBDown（可学习双边下采样）**：替换 backbone 前两级 `Conv [64,3,2]` / `Conv [128,3,2]`。
  在 2x2 邻域取 4 个采样点做加权和实现 1/2 下采样，权重 = 可学习空间先验 × 强度相似度；
  采样点的亚像素偏移由 3x3 卷积预测（`tanh` 限幅），强度响应由 1x1 卷积 + sigmoid 学习，
  高斯带宽 `sigma` 为可学习参数。用 `F.grid_sample` 完成可微采样。
- **CGBlockAttn（粗粒度块注意力）**：插在 backbone P3、P4 级各一个。
  按 8x8 不重叠块做平均池化得到粗粒度 token（token 数 ≈ HW/64），在 token 上做 4 头自注意力，
  再广播回像素、经逐像素门控（`sigmoid(conv(x))`）残差相加。注意力用显式 `matmul + softmax`
  实现，而非 `nn.MultiheadAttention`，便于 ONNX 导出。

注册点：`ultralytics/nn/modules/__init__.py`（导入 + `__all__`）、`ultralytics/nn/tasks.py`
（导入 + `parse_model` 的 `base_modules`，从而自动处理 `c1/c2` 与 width 缩放）。

## 3. 结构开销（imgsz=640，fp32，batch=1）

| 配置                   | 参数量          | GFLOPs      | PyTorch 延迟    | 峰值显存        |
| ---------------------- | --------------- | ----------- | --------------- | --------------- |
| baseline `yolo26n`     | 2.572 M         | 6.12        | 19.72 ms        | 83.5 MiB        |
| `+LBDown +CGBlockAttn` | 2.669 M (+3.8%) | 7.04 (+15%) | 27.45 ms (+39%) | 95.8 MiB (+15%) |

## 4. TensorRT 实测（trtexec `--loadEngine`，200 iters，GPU Compute Time 中位数）

| engine                | 精度 | GPU 延迟    | 相对基线                          |
| --------------------- | ---- | ----------- | --------------------------------- |
| `base-fp32` `yolo26n` | fp32 | **2.05 ms** | 1.00x                             |
| `cgba-fp32`           | fp32 | **2.88 ms** | 1.41x                             |
| `base-q16`            | fp16 | **1.07 ms** | 1.00x（相对 fp32 **1.92x 加速**） |
| `cgba-q16`            | fp16 | **1.65 ms** | 1.55x                             |

ONNX 规模：base 9.4 MiB(fp32) / 4.8 MiB(fp16)；cgba 9.8 MiB(fp32) / 5.0 MiB(fp16)。

## 5. 数值一致性（TensorRT vs PyTorch，同一输入）

| 配置      | max\|Δ\| | max relative |
| --------- | -------- | ------------ |
| base fp32 | 3.05e-05 | 1.55e-06     |
| cgba fp32 | 1.83e-04 | 7.51e-06     |
| base fp16 | 1.53e-05 | 1.92e-03     |
| cgba fp16 | 1.22e-04 | 1.92e-03     |

结论：LBDown 的 `grid_sample` 让 TRT 与 PyTorch 的偏差放大约 6 倍（3e-5 → 1.8e-4），
量级仍可忽略，但说明**非标准算子必须做一致性比对，不能只看"跑得通"**。

## 6. 实测踩到的真实困难

1. **pip 版 TensorRT 与 Python 版本不匹配**：Python 3.14 下没有可用的 `tensorrt-cu12` wheel，
   改为下载完整 SDK 用 `trtexec` 命令行构建，代价是构建流程脱离 Python、参数要自己管。
2. **TensorRT 11 取消了构建期精度开关**：`--fp16` / `--int8` 在 TRT 11 的 `trtexec` 中已不存在
   （只剩 `--noTF32`，`--stronglyTyped` 已废弃为 no-op），网络默认强类型化 —— 精度必须在
   **ONNX 导出阶段**决定。
3. **ultralytics 的导出参数也在变**：`half=True` 已弃用，改为 `quantize=16`；沿用旧写法会得到
   过时告警甚至错误结果。
4. **自定义结构要改 yaml 索引**：新增层会让 head 里所有 `Concat` 的 `from` 索引顺延
   （原 6/4/13/10 → 8/5/15/12），漏改会导致通道拼接错误。
5. **`grid_sample` 的部署可用性**：导出的 ONNX 含 16 个 `GridSample` 节点，TRT 11 实测可解析
   通过（无需 plugin），但必须验证数值一致性（见第 5 节）。
6. **FLOPs 不能代表延迟**：结构只增加 15% GFLOPs，延迟却增加 41%（fp32）/ 55%（fp16）。
   瓶颈不在计算量而在于 `grid_sample` 这类访存型算子，说明轻量化改造必须实测端到端延迟，
   不能只看 FLOPs 指标。

## 7. 复现命令

```bash
conda run -n trt python check_cgba.py # 结构自检 + 参数量/延迟
conda run -n trt python bench_trt.py --cfg ultralytics/cfg/models/26/yolo26-cgba.yaml --tag cgba
conda run -n trt python bench_trt.py --cfg ultralytics/cfg/models/26/yolo26.yaml --tag base
# fp16 版本：追加 --quantize 16
```

## 8. 尚未验证（不要对外声称）

- **精度/召回指标**：本机无航拍数据集，未做 DroneVehicle 训练与消融，**没有任何 mAP 提升数据**。
- **跨设备可用性**：engine 与 TRT 版本/GPU 架构绑定，未在 Jetson 等边缘设备验证。
- **动态 shape / INT8**：均未测试。
