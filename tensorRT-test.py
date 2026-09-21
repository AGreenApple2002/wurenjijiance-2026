# Pytorch转ONNX
# from ultralytics import YOLO
# model = YOLO("yolo26n.pt")
# model.export(format="onnx", imgsz=640, simplify=True, opset=17)

#推理ONNX模型
# yolo predict model=yolo26n.onnx


# ONNX转TensorRT 使用pip版的tensorRT工具
# from ultralytics import YOLO
# m = YOLO("yolo26n.pt")
# m.export(format="engine", imgsz=640, half=True, workspace=4)  # 生成 yolo26n.engine

# 因为我的python版本是3.14，没有匹配的pip版本。于是我转为使用下载完整SDK的方式，安装trtexec工具，在命令行中转换模型。
#TRT=<TensorRT_SDK_ROOT>   # 例如 /opt/TensorRT-11.0.0.114
# export LD_LIBRARY_PATH="$TRT/lib:$LD_LIBRARY_PATH"
# cd <项目根目录>          # 例如 ultralytics-main
# # ONNX → TensorRT(FP32)
# "$TRT/bin/trtexec" --onnx=yolo26n.onnx --saveEngine=yolo26n-fp32.engine


# # 推理TensorRT模型
from ultralytics import YOLO
trt = YOLO("yolo26n-my.engine")
results = trt.predict("https://ultralytics.com/images/bus.jpg", save=True)