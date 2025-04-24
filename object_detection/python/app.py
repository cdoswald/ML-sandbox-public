"""Real-time object detection with YOLOv11."""

import cv2
import PIL
import matplotlib.pyplot as plt
import matplotlib as mpl

from ultralytics import YOLO

image_path = "images/cat.jpeg"
pretrained_model = "yolo11n.pt"

model = YOLO(pretrained_model)

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image)[0]

fig, ax = plt.subplots()
ax.imshow(image)
for r in results.boxes:
    label = model.names.get(int(r.cls), None)
    conf = float(r.conf)
    xywh = r.xywh.flatten()
    w = float(xywh[2])
    h = float(xywh[3])
    x0 = float(xywh[0]) - w/2
    y0 = float(xywh[1]) - h/2

    bbox = mpl.patches.Rectangle(
        (x0, y0), w, h,
        fill=False, edgecolor="red", lw=1.5,
    )
    ax.add_patch(bbox)
    plot_label = f"{label} {conf:.2f}"
    ax.text(
        x, y-5, plot_label, fontsize=8, color="white",
        bbox=dict(facecolor="red", alpha=0.5, pad=2))
    ax.axis("off")
