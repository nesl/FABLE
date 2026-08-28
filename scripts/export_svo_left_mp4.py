#!/usr/bin/env python3
"""Export the left image stream of one SVO/SVO2 as an H.264 MP4.

This program runs inside the repository's ZED SDK image.  Host-side callers
must validate/cache its output before physical replay staging.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import pyzed.sl as sl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.svo_real_time_mode = False
    init.set_from_svo_file(str(source))
    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"could not open SVO: {status}")
    try:
        info = camera.get_camera_information().camera_configuration
        width = int(info.resolution.width)
        height = int(info.resolution.height)
        fps = max(1, int(round(float(info.fps))))
        image = sl.Mat(width, height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
        runtime = sl.RuntimeParameters()
        encoder = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s:v", f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ],
            stdin=subprocess.PIPE,
        )
        frames = 0
        while True:
            grabbed = camera.grab(runtime)
            if grabbed == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if grabbed != sl.ERROR_CODE.SUCCESS:
                continue
            camera.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
            # PyZED returns BGRA; FFmpeg is explicitly configured for BGR24.
            encoder.stdin.write(image.get_data()[:, :, :3].tobytes())
            frames += 1
        encoder.stdin.close()
        returncode = encoder.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with {returncode}")
        if frames == 0:
            raise RuntimeError("SVO contained no decodable left-camera frames")
        print(f"frames={frames} fps={fps} width={width} height={height}")
    finally:
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
