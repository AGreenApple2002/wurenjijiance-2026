from ultralytics import YOLO

model = YOLO("/home/guiyan/ztz/repo/ultralytics-main/runs/detect/DroneVehicle_train100/weights/best.pt")
metrics = model.val(data="DroneVehicle.yaml", split="val", device=0)
