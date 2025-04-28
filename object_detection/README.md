## Object Detection with YOLO

This directory contains object detection implementations in [Python](object_detection/python) and [C++](object_detection/cpp). 

Both implementations use [OpenCV](https://opencv.org/) for video processing and the [YOLO11](https://docs.ultralytics.com/models/yolo11/) model from Ultralytics for object detection. For inference, the Python implementation uses PyTorch while the C++ implementation uses an ONNX Runtime.

Each implementation has separate Docker and config files. To run the application, specify the input video, output video, and model paths in the config file. There is also a parameter to display inference in real-time.

(NOTE: the applications have only been tested on `.mp4` videos shot on iPhone 14)

<img src="object_detection/ex_frame.jpg" width="25%">

