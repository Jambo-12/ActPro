"""
Prepare TMM revision videos with aspect-ratio preserving letterbox resize.

This keeps the full camera view: the longer side is resized to 384 and the
shorter side is padded with black to produce 384x384 videos. It is intentionally
separate from the person-centered crop384 preprocessing.

Example, from Act-LLaVA root:
  python -m data.preprocess.letterbox_resize_tmm \
      --src /path/to/TMM_revision/videos \
      --dst dataset/TMM_revision_letterbox384/videos_2fps/test \
      --fps 2 --resolution 384
"""
import argparse
import os
import subprocess

from data.utils import get_ffmpeg_bin, list_videos


def ffmpeg_letterbox_once(src_path, dst_path, fps, resolution):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        get_ffmpeg_bin(),
        "-y",
        "-loglevel",
        "error",
        "-sws_flags",
        "bicubic",
        "-i",
        src_path,
        "-an",
        "-threads",
        "10",
        "-r",
        str(fps),
        "-vf",
        (
            f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':"
            f"'if(gt(iw\\,ih)\\,-2\\,{resolution})',"
            f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='#000000'"
        ),
        dst_path,
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Letterbox-resize TMM videos to square 384x384.")
    parser.add_argument("--src", required=True, help="Source video directory.")
    parser.add_argument("--dst", required=True, help="Output video directory.")
    parser.add_argument("--fps", type=int, default=2, help="Target FPS.")
    parser.add_argument("--resolution", type=int, default=384, help="Target square size.")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    videos = sorted(list_videos(args.src))
    print(
        f"[letterbox_resize_tmm] {len(videos)} videos: "
        f"{args.src} -> {args.dst} @ {args.fps}fps, {args.resolution}x{args.resolution}"
    )

    done = skipped = failed = 0
    for idx, src_path in enumerate(videos, start=1):
        key = os.path.splitext(os.path.basename(src_path))[0]
        dst_path = os.path.join(args.dst, key + ".mp4")
        if os.path.exists(dst_path):
            skipped += 1
            print(f"  [{idx}/{len(videos)}] skip {key}.mp4")
            continue
        try:
            ffmpeg_letterbox_once(src_path, dst_path, args.fps, args.resolution)
            done += 1
            print(f"  [{idx}/{len(videos)}] wrote {key}.mp4")
        except Exception as exc:
            failed += 1
            print(f"  [{idx}/{len(videos)}] FAILED {key}: {exc}")

    print(f"[letterbox_resize_tmm] done={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
