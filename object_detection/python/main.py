"""Real-time object detection with YOLOv11."""

import cv2
import json
import PIL
import re
import subprocess
import time
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl

from ultralytics import YOLO

# Define constants
FRAME_ROTATION_CODES = {
    -90: cv2.ROTATE_90_CLOCKWISE,
    90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    180: cv2.ROTATE_180,
    -180: cv2.ROTATE_180,
}

# Define functions
def check_video_rotation(filepath: str) -> Optional[int]:
    """Check video file for rotation metadata."""
    cmd = [
        "ffprobe", 
        "-show_entries", "side_data",
        "-print_format", "json",
        filepath
    ]
    metadata = subprocess.check_output(cmd).decode("utf-8")
    rot_match = re.search(r'"rotation":\s*-?\d+', metadata)
    if rot_match:
        return int(rot_match.group().replace('"rotation": ', ''))
    return


def annotate_video(
    input_path: str,
    output_path: str,
    YOLO_model_path: str = "yolo11n.pt",
    real_time_display: bool = False,
) -> None:
    """Detect objects in input video and save annotated output video."""

    start_time = time.time()
    if real_time_display:
        break_key = "q"
        print(f"Displaying video in real-time. Press {break_key} to break early.")

    # Load YOLO model
    model = YOLO(YOLO_model_path)

    # Load input video
    cap = cv2.VideoCapture(input_path)

    # Check if original video is rotated (e.g., iPhone camera)
    frame_rot_deg = check_video_rotation(input_path)
    if frame_rot_deg is not None:
        try:
            rotation_code = FRAME_ROTATION_CODES[frame_rot_deg]
        except KeyError:
            raise ValueError(
                f"Frame rotation must be in {-90, 90, 180}, but got {frame_rot_deg}."
            )

    # Create output video writer
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    frame_size = (int(cap.get(3)), int(cap.get(4)))
    if frame_rot_deg in [-90, 90]:
        frame_size = (frame_size[1], frame_size[0])
    output = cv2.VideoWriter(output_path, fourcc, 30.0, frame_size)

    # Run model
    while cap.isOpened():
        retval, frame = cap.read()
        if not retval:
            break
        
        # Preprocess image
        # (note that rotation is permanent change whereas RGB is temporary for YOLO model)
        if frame_rot_deg is not None:
            frame = cv2.rotate(frame, rotation_code) 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Generate predictions
        results = model(frame_rgb)[0]

        for r in results.boxes:
            label = model.names.get(int(r.cls))
            conf = float(r.conf)
            x1, y1, x2, y2 = map(int, r.xyxy.flatten())
            
            # Draw bounding boxes
            rec_color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), rec_color)

            # Add label
            text = f"{label} {conf:.2f}"
            text_pos = (x1, y1 - 10 if y1 - 10 > 10 else y1 + 10)
            (font_width, font_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_box_coords = (x1, y1 - font_height - 10), (x1 + font_width, y1)
            cv2.rectangle(frame, label_box_coords[0], label_box_coords[1], rec_color, -1)
            cv2.putText(
                frame,
                text,
                text_pos,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        # Write to output file
        output.write(frame)

        # Display in real-time (if applicable)
        if real_time_display:
            cv2.imshow("Annotated Video", frame)
        if cv2.waitKey(1) & 0xFF == ord(break_key):
            break

    # Clean-up
    cap.release()
    output.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"Total annotation time: {end_time - start_time:.3f} seconds")


def main():
    """Load config and annotate video."""
    with open("config.json", "r") as file:
        config = json.load(file)
    annotate_video(
        input_path=config["input_path"],
        output_path=config["output_path"],
        YOLO_model_path=config["YOLO_model_path"],
        real_time_display=config["real_time_display"],
    )


if __name__ == "__main__":
    main()
