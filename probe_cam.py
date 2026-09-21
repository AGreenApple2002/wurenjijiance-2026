# quick camera probe: try MJPG + explicit size, with retries
import sys

import cv2

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("opened:", cap.isOpened())
print(
    "fourcc:",
    int(cap.get(cv2.CAP_PROP_FOURCC)),
    "size:",
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    "x",
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
)

got = False
for i in range(15):
    ret, f = cap.read()
    print(f"attempt {i}: ret={ret}")
    if ret:
        print("FRAME OK shape=", f.shape)
        got = True
        break

# try dumping a few more frames to see if stream is live
if got:
    n = 0
    for i in range(20):
        ret, f = cap.read()
        if ret:
            n += 1
    print(f"stream: {n}/20 frames after first")
cap.release()
print("RESULT:", "OK" if got else "NO_FRAME")
