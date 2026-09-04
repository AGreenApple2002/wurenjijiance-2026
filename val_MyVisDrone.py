from ultralytics import YOLO

# 加载权重
model = YOLO("/home/guiyan/ztz/repo/ultralytics-main/runs/detect/MyVisDrone_train100/weights/best.pt")

# 在 MyVisDrone.yaml 上做验证（CPU）
metrics = model.val(
    data="MyVisDrone.yaml",
    split="val",
    imgsz=640,
    device=0
)
