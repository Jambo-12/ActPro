"""Prepare PKUMMD videos for the released ActLLaVA-PKUMMD setup.

The released PKUMMD model uses videos sampled to 2 FPS, center-cropped to the
largest square region, and resized to 384x384 before SigLIP feature extraction.

This script expects PKUMMD video basenames to be unique, such as
`0291-L.avi` or `0291-L.mp4`. It writes a flat output directory whose basenames
match the PKUMMD annotation keys.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tqdm

from data.utils import ffmpeg_once, list_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="PKUMMD 2FPS center-crop 384 preprocessing.")
    parser.add_argument("--src", required=True, help="Raw PKUMMD video directory.")
    parser.add_argument("--dst", required=True, help="Flat output directory for processed videos.")
    parser.add_argument("--fps", type=int, default=2, help="Target frame rate.")
    parser.add_argument("--resolution", type=int, default=384, help="Output square resolution.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate videos that already exist.")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(list_videos(str(src_root)))
    print(
        f"[*] {len(videos)} videos: {src_root} -> {dst_root} "
        f"@ {args.fps} FPS, center crop, {args.resolution}x{args.resolution}"
    )

    seen: set[str] = set()
    done = skipped = failed = 0
    for src in tqdm.tqdm(videos):
        src_path = Path(src)
        dst_name = src_path.with_suffix(".mp4").name
        if dst_name in seen:
            raise RuntimeError(f"Duplicate PKUMMD basename after flattening: {dst_name}")
        seen.add(dst_name)
        dst_path = dst_root / dst_name
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            ffmpeg_once(
                str(src_path),
                str(dst_path),
                fps=args.fps,
                resolution=args.resolution,
                crop_center=True,
            )
            done += 1
        except Exception as exc:  # noqa: BLE001 - keep batch preprocessing moving.
            failed += 1
            print(f"Error processing {src_path}: {exc}")
            if dst_path.exists():
                os.remove(dst_path)

    print(f"[*] done={done} skipped={skipped} failed={failed} -> {dst_root}")


if __name__ == "__main__":
    main()
