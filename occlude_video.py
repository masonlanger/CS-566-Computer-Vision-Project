import cv2
import numpy as np

def run(input_path, output_path, rx=None, ry=None, rw=None, rh=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Defaults if not provided
    x = 0 if rx is None else int(rx)
    y = 0 if ry is None else int(ry)
    w = width if rw is None else int(rw)
    h = height if rh is None else int(rh)

    # Clamp to frame bounds and handle negatives
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, w)
    h = max(0, h)
    if x + w > width:
        w = width - x
    if y + h > height:
        h = height - y

    if w == 0 or h == 0:
        print("Warning: blackout rectangle has zero area; nothing to mask.")

    # Write out using mp4 container with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open output for writing: {output_path}")

    start_frame = 0
    end_frame = total_frames

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_idx <= end_frame and w > 0 and h > 0:
            frame[y:y+h, x:x+w] = 0  # blackout rectangle

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved video to {output_path}. Masked rect: (x={x}, y={y}, w={w}, h={h}).")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Black out a rectangle in an MP4 video.")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("output", help="Path to output video")
    parser.add_argument("--x", "-x", type=int, default=None, help="Rectangle top-left x (default: 0)")
    parser.add_argument("--y", "-y", type=int, default=None, help="Rectangle top-left y (default: 0)")
    parser.add_argument("--width", "-W", type=int, default=None, help="Rectangle width (default: video width)")
    parser.add_argument("--height", "-H", type=int, default=None, help="Rectangle height (default: video height)")
    args = parser.parse_args()

    run(args.input, args.output, args.x, args.y, args.width, args.height)
