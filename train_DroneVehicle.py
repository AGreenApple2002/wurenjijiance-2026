from ultralytics import YOLO

model = YOLO("yolo26x.pt")

results = model.train(
    data="DroneVehicle.yaml",
    epochs=100,
    imgsz=640,
    device=[0, 1],
    name="DroneVehicle_train100"
)
