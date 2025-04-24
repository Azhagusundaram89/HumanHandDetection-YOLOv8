from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="D:\hand detection\hand-1\data.yaml", epochs=100, imgsz=640)
