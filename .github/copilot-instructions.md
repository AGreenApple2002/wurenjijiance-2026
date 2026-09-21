# Copilot / 智能体会话说明（本地）

本仓库是 ultralytics 的**本地实验克隆**，被改为无人机航拍小目标检测的结构实验工程。
其中包含**非上游的自定义改动**，接手前请先读：

- **`CGBA_GUIDE.md`** —— 改动索引（文件 + 行号）、模块设计、yaml 接线与索引顺延、运行命令、实测数据、踩坑清单
- **`EXPERIMENT_NOTES.md`** —— 实验记录、TRT 部署困难清单

## 自定义改动速查

| 类型     | 路径                                                                                                                      |
| -------- | ------------------------------------------------------------------------------------------------------------------------- |
| 新增模块 | `ultralytics/nn/modules/cgba.py`（`LBDown` 可学习双边下采样、`CGBlockAttn` 粗粒度块注意力）                               |
| 新增配置 | `ultralytics/cfg/models/26/yolo26-cgba.yaml`                                                                              |
| 注册点   | `ultralytics/nn/modules/__init__.py`（L63/L120/L168）、`ultralytics/nn/tasks.py`（L44/L61、L2068-2069 的 `base_modules`） |
| 实验脚本 | `check_cgba.py`、`bench_trt.py`、`export_cgba_onnx.py`、`tensorRT-test.py`、`trt_infer_pure.py`、`trt_webcam.py`          |

## 环境

- conda 环境 `trt`：Python 3.12.14、torch 2.6.0+cu124、tensorrt 11.0.0.114
- TensorRT SDK：自行下载解压到任意目录（`bin/trtexec`），使用前需设置 `LD_LIBRARY_PATH="$TRT_ROOT/lib"`
- GPU：RTX 4060 Laptop 8 GB

## 硬性约束

1. 不要报出精度 / mAP / 召回数字：**没有训练、没有数据集、没有任何精度数据**。
2. 修改 backbone/head 层结构后，必须同步顺延 `yolo26-cgba.yaml` 中 head 里所有 `Concat` 的 `from` 索引。
3. 新增模块必须同时注册到 `modules/__init__.py` 与 `tasks.py` 的 `base_modules`，否则无法吃到 width 缩放。
4. 结构改动后重跑 `check_cgba.py` 与 `bench_trt.py`，并更新 `CGBA_GUIDE.md` 第 6 节数据。
5. `AGENTS.md` 与 `CLAUDE.md`（软链）保留的是上游 ultralytics 规范，本文件与 `CGBA_GUIDE.md` 是本地增量。
