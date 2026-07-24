"""Downsample ASTime videos to a target FPS while keeping resolution.

For the released main ASTime model, this script is used with `--fps 2`.
It only changes frame rate. It does not resize or crop; spatial processing is
handled later by the LLaVA-OneVision/SigLIP image processor during feature
extraction.

The source directory may be nested. The output directory is flat, and each
output file is named `<basename>.mp4` so the basename matches the annotation key
and feature tensor name.

Example, from the Act-LLaVA root:
    python -m data.preprocess.downsample_astime_fps \
        --src /path/to/ASTime/raw/train/videos \
        --dst dataset/ASTime/videos_2fps --fps 2
"""
import argparse
import os

from data.utils import ffmpeg_once, list_videos


def main():
    ap = argparse.ArgumentParser(description="Downsample ASTime videos to a target fps (keep resolution).")
    ap.add_argument('--src', required=True, help='Source video root directory. Nested folders are allowed.')
    ap.add_argument('--dst', required=True, help='Flat output directory for sampled .mp4 files.')
    ap.add_argument('--fps', type=int, required=True, help='Target frame rate.')
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    videos = sorted(list_videos(args.src))
    print(f'[*] {len(videos)} videos: {args.src} -> {args.dst} @ {args.fps} FPS (keep resolution)')

    done, skipped, failed = 0, 0, 0
    for i, src in enumerate(videos):
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(args.dst, base + '.mp4')
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            ffmpeg_once(src, dst, fps=args.fps, resolution=None)
            done += 1
            print(f'[{i + 1}/{len(videos)}] {base}.mp4')
        except Exception as e:
            failed += 1
            print(f'[{i + 1}/{len(videos)}] FAILED {base}: {e}')

    print(f'[*] done={done} skipped(exist)={skipped} failed={failed} -> {args.dst}')


if __name__ == '__main__':
    main()
