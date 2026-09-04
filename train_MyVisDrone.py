from ultralytics import YOLO

model = YOLO("/home/guiyan/ztz/repo/ultralytics-main/runs/detect/DroneVehicle_train100/weights/best.pt")

results = model.train(
    data="MyVisDrone.yaml",
    epochs=100,
    imgsz=640,
    device=0,
    name="MyVisDrone_train100"
)
