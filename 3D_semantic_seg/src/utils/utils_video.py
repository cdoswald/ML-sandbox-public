"""Utility functions for video processing."""

from typing import Optional

import ffmpeg


def vstack_videos(
    video_file_1: str,
    video_file_2: str,
    output_file: str,
    output_resolution: tuple[int, int] = (1920, 1080),
    video1_crop_height: Optional[int] = None,
    video1_crop_y_offset: Optional[int] = None,
    video2_crop_height: Optional[int] = None,
    video2_crop_y_offset: Optional[int] = None,
) -> None:
    """Stitch two videos together vertically.

    Args:
        video_file_1: string file path for first video
        video_file_2: string file path for second video
        output_file: string file path for stitched output video

    Returns:
        None (saves stitched video to output_file)
    """
    try:
        # probe = ffmpeg.probe(video_file_1)
        # video1_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        # video1_width = int(video1_stream['width'])
        # video1_height = int(video1_stream['height'])
        # print(f"File Dimensions: {video1_width}x{video1_height}")

        # probe = ffmpeg.probe(video_file_2)
        # video2_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        # video2_width = int(video2_stream['width'])
        # video2_height = int(video2_stream['height'])
        # print(f"File Dimensions: {video2_width}x{video2_height}")

        video1 = ffmpeg.input(video_file_1)
        video2 = ffmpeg.input(video_file_2)
        # Scale videos to have same width
        video1 = video1.filter("scale", output_resolution[0], -1)
        video2 = video2.filter("scale", output_resolution[0], -1)
        # Crop videos (if applicable)
        if video1_crop_height is not None and video1_crop_y_offset is not None:
            video1 = video1.filter(
                "crop",
                output_resolution[0],
                video1_crop_height,
                0,
                video1_crop_y_offset,
            )
        if video2_crop_height is not None and video2_crop_y_offset is not None:
            video2 = video2.filter(
                "crop",
                output_resolution[0],
                video2_crop_height,
                0,
                video2_crop_y_offset,
            )
        # Stack videos
        combined = ffmpeg.filter([video1, video2], "vstack")
        output = ffmpeg.output(combined, output_file)
        ffmpeg.run(output, overwrite_output=True)
    except ffmpeg.Error as e:
        print(f"Error stitching videos '{video_file_1}' and '{video_file_2}'")
        raise
