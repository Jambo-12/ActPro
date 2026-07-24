"""Single-GPU feature extraction without Submitit.

`extract_feature.py` uses Submitit's local executor. On some machines, GPU
placement can ignore the outer CUDA_VISIBLE_DEVICES setting and send multiple
jobs to the same GPU. This script runs directly on the selected local GPU.

The encoding logic matches `extract_feature.py`: full aspect ratio, float16
processed images, `image_token_generation(..., batch_size=16)`, and bfloat16
saved tensors.

Example, from the Act-LLaVA root:
    CUDA_VISIBLE_DEVICES=0 python -m data.preprocess.extract_feature_simple \
        --video_dir dataset/ASTime/videos_2fps \
        --output_dir dataset/ASTime/features
"""
import argparse
import os

import torch
import torchvision
import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, image_token_generation


def encode_frames(frames, image_processor, model_config, vision_tower, device):
    frames_tensor = process_images(frames, image_processor, model_config)
    frames_tensor = [f.to(dtype=torch.float16, device=device) for f in frames_tensor]
    with torch.no_grad():
        features = image_token_generation(vision_tower=vision_tower, images=frames_tensor, batch_size=16)
    return features.to(dtype=torch.bfloat16).to(device='cpu')


def encode_video_chunked(video_path, image_processor, model_config, vision_tower, device, read_chunk_size):
    reader = torchvision.io.VideoReader(video_path, "video")
    chunks, frames = [], []
    for item in reader:
        frames.append(item["data"])
        if len(frames) >= read_chunk_size:
            chunks.append(encode_frames(torch.stack(frames), image_processor, model_config, vision_tower, device))
            frames.clear()
    if frames:
        chunks.append(encode_frames(torch.stack(frames), image_processor, model_config, vision_tower, device))
    if not chunks:
        raise RuntimeError("No frames decoded")
    return torch.cat(chunks, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Single-GPU feature extraction (no submitit).")
    ap.add_argument('--video_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--pretrained', default='lmms-lab/llava-onevision-qwen2-7b-ov')
    ap.add_argument('--gpu', type=int, default=0, help='Local GPU index relative to CUDA_VISIBLE_DEVICES, usually 0.')
    ap.add_argument('--read_chunk_size', type=int, default=0,
                    help='Read frames in chunks to avoid high memory use on long videos; 0 keeps one-shot read_video.')
    args = ap.parse_args()

    device = f'cuda:{args.gpu}'
    _, model, image_processor, _ = load_pretrained_model(
        args.pretrained, None, 'llava_qwen', device_map={"": device}, attn_implementation="sdpa")
    model.eval()
    setattr(model.config, "image_aspect_ratio", "full")
    vision_tower = model.get_vision_tower()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(os.listdir(args.video_dir))
    done, skipped, failed = 0, 0, 0
    for file in tqdm.tqdm(files, desc=f'{args.video_dir} -> {args.output_dir}'):
        video_path = os.path.join(args.video_dir, file)
        save_path = os.path.join(args.output_dir, os.path.splitext(file)[0] + '.pt')
        if os.path.exists(save_path):
            skipped += 1
            continue
        try:
            if args.read_chunk_size > 0:
                features = encode_video_chunked(
                    video_path, image_processor, model.config, vision_tower, device, args.read_chunk_size)
            else:
                frames = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
                features = encode_frames(frames, image_processor, model.config, vision_tower, device)
            torch.save(features, save_path)
            done += 1
        except Exception as e:
            failed += 1
            print(f"Error processing {video_path}: {e}")

    print(f'[*] done={done} skipped={skipped} failed={failed} -> {args.output_dir} '
          f'({len([x for x in os.listdir(args.output_dir) if x.endswith(".pt")])} .pt total)')


if __name__ == '__main__':
    main()
