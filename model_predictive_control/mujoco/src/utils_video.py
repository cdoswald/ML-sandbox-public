import cv2


def display_video_from_frame_list(frame_list, frame_delay=33):
    """
    Display a video from a list of frames using OpenCV.

    Args:
        frame_list (list): List of frames (images) to display.
        frame_delay (int): Delay between frames in milliseconds. Default is 33ms (~30 FPS).
    """
    window_name = "Video Playback"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame in frame_list:
        if frame.shape[2] == 4:
            cv2_color_conv_code = cv2.COLOR_RGBA2BGRA
        else:
            cv2_color_conv_code = cv2.COLOR_RGB2BGR
        cv2.imshow(window_name, cv2.cvtColor(frame, cv2_color_conv_code))

        # Exit on 'q' key press
        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break

    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
