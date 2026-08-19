import cv2
import numpy as np
import time


def open_camera(device_path: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
	cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
	if not cap.isOpened():
		# Some devices work better with OpenCV's default backend fallback.
		cap.release()
		cap = cv2.VideoCapture(device_path)
	if cap.isOpened():
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		cap.set(cv2.CAP_PROP_FPS, fps)
	return cap


def unavailable_frame(width: int, height: int, label: str) -> np.ndarray:
	frame = np.zeros((height, width, 3), dtype=np.uint8)
	cv2.putText(
		frame,
		f"No signal: {label}",
		(20, height // 2),
		cv2.FONT_HERSHEY_SIMPLEX,
		1.0,
		(0, 0, 255),
		2,
		cv2.LINE_AA,
	)
	return frame


class CameraStream:
	def __init__(self, device_path: str, width: int, height: int, fps: int) -> None:
		self.device_path = device_path
		self.width = width
		height = height
		self.height = height
		self.fps = fps
		self.cap: cv2.VideoCapture | None = None
		self.last_open_attempt = 0.0
		self.reconnect_interval_s = 0.75
		self.open()

	def open(self) -> None:
		if self.cap is not None:
			self.cap.release()
		self.cap = open_camera(self.device_path, self.width, self.height, self.fps)
		self.last_open_attempt = time.monotonic()

	def read_frame(self) -> np.ndarray:
		if self.cap is None or not self.cap.isOpened():
			if time.monotonic() - self.last_open_attempt >= self.reconnect_interval_s:
				self.open()
			return unavailable_frame(self.width, self.height, self.device_path)

		ok, frame = self.cap.read()
		if ok and frame is not None:
			return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

		# When unplug/replug happens, read can fail while the handle still looks open.
		self.cap.release()
		self.cap = None
		return unavailable_frame(self.width, self.height, self.device_path)

	def close(self) -> None:
		if self.cap is not None:
			self.cap.release()
			self.cap = None


def main() -> None:
	stream_width = 640
	stream_height = 480
	stream_fps = 30

	left_device = "/dev/video0"
	right_device = "/dev/video2"

	left_stream = CameraStream(left_device, stream_width, stream_height, stream_fps)
	right_stream = CameraStream(right_device, stream_width, stream_height, stream_fps)

	window_name = "Stereo Streams (/dev/video0 | /dev/video2)"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	try:
		while True:
			left_frame = left_stream.read_frame()
			right_frame = right_stream.read_frame()

			combined = np.hstack((left_frame, right_frame))
			cv2.imshow(window_name, combined)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break
	finally:
		left_stream.close()
		right_stream.close()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
