"""
ASTime 自适应人体裁剪预处理(为与 PKUMMD 统一、做联合训练用)。

与现有 ASTime/PKUMMD 的预处理【完全隔离】:全新脚本 + 独立输出路径,不改动也不依赖
任何现有的训练/测试/特征提取流程。

做的事(每个视频):
  1. decord 读原视频,按时间戳采样到 2fps(输入是 2fps 还是 30fps 都可,统一变 2fps);
  2. torchvision Faster R-CNN 逐帧检测人(COCO person=1),取主要人物(置信度×面积最大)
     的水平中心 x;
  3. 对 x 轨迹做缺失插值 + 时间平滑(去抖),保证窗口跟着人平滑移动;
  4. 满高正方形(边长=画面高 H)按平滑后的 x 自适应水平裁剪(越界贴边),再缩放到 384×384;
  5. 用 2fps 写出裁剪后的视频。
  无人帧 -> 插值/邻近;整段无人 -> 退化为中心裁剪。

用法(在 Act-LLaVA/ 根目录):
  python -m data.preprocess.adaptive_crop_astime \
      --input_dir /path/to/ASTime/2fps/train/videos \
      --output_dir dataset/ASTime_crop384/videos_2fps/train --gpu 0
  python -m data.preprocess.adaptive_crop_astime \
      --input_dir dataset/ASTime/videos/test \
      --output_dir dataset/ASTime_crop384/videos_2fps/test --gpu 0
"""
import os
import argparse
import numpy as np
import cv2
import torch
import decord
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
)

decord.bridge.set_bridge('native')

TARGET_FPS = 2
OUT_SIZE = 384
PERSON_LABEL = 1          # COCO: person
DET_LONG_SIDE = 960       # 检测前把帧缩到长边 960,加速(框再按比例还原)
VIDEO_EXTS = ('.mp4', '.MP4', '.avi', '.AVI', '.mov', '.MOV')


def list_videos(d):
    out = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(VIDEO_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def sample_indices_2fps(num_frames, src_fps):
    """按 0.5s 间隔取帧索引,实现 2fps 采样(输入已是 2fps 时≈全取)。"""
    duration = num_frames / src_fps
    times = np.arange(0.0, duration, 1.0 / TARGET_FPS)
    idxs = np.clip(np.round(times * src_fps).astype(int), 0, num_frames - 1)
    return idxs


@torch.no_grad()
def detect_person_centers(frames_rgb, model, device, conf=0.5, batch=8):
    """对一批 RGB 帧(uint8, NHWC)逐帧检测主要人物水平中心 x(原图坐标);无人返回 NaN。"""
    H, W = frames_rgb.shape[1], frames_rgb.shape[2]
    scale = DET_LONG_SIDE / max(H, W)
    dw, dh = int(round(W * scale)), int(round(H * scale))
    centers = np.full(len(frames_rgb), np.nan, dtype=np.float32)
    for s in range(0, len(frames_rgb), batch):
        chunk = frames_rgb[s:s + batch]
        tensors = []
        for fr in chunk:
            small = cv2.resize(fr, (dw, dh), interpolation=cv2.INTER_LINEAR)
            tensors.append(torch.from_numpy(small).permute(2, 0, 1).float().div(255.0).to(device))
        outputs = model(tensors)
        for j, o in enumerate(outputs):
            keep = (o['labels'] == PERSON_LABEL) & (o['scores'] >= conf)
            if keep.any():
                boxes = o['boxes'][keep]
                scores = o['scores'][keep]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                best = int(torch.argmax(scores * areas))   # 置信度×面积 最大者=主要人物
                cx_small = float((boxes[best, 0] + boxes[best, 2]) / 2)
                centers[s + j] = cx_small / scale            # 还原到原图坐标
    return centers


def fill_and_smooth(centers, W, window=7):
    """缺失插值 + 滑动中位平滑;整段无人 -> 画面中心。"""
    x = centers.copy()
    valid = ~np.isnan(x)
    if not valid.any():
        return np.full(len(x), W / 2.0, dtype=np.float32)
    idx = np.arange(len(x))
    x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])  # 线性插值补缺
    # 滑动中位去抖
    half = window // 2
    sm = x.copy()
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        sm[i] = np.median(x[lo:hi])
    return sm


def crop_one(frame_rgb, cx, side, out_size):
    """满高正方形(边=side=H),水平中心对准 cx(贴边),缩放到 out_size。"""
    H, W = frame_rgb.shape[:2]
    x0 = int(round(cx - side / 2))
    x0 = max(0, min(x0, W - side))            # 贴边,保证窗口在画面内
    crop = frame_rgb[:, x0:x0 + side]          # 满高
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def process_video(path, out_path, model, device, conf, window):
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    src_fps = vr.get_avg_fps()
    n = len(vr)
    idxs = sample_indices_2fps(n, src_fps)
    frames = vr.get_batch(list(idxs)).asnumpy()     # (T,H,W,3) RGB uint8
    H, W = frames.shape[1], frames.shape[2]
    side = min(H, W)                                 # 满高正方形边长

    centers = detect_person_centers(frames, model, device, conf=conf)
    n_det = int((~np.isnan(centers)).sum())
    cx = fill_and_smooth(centers, W, window=window)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             TARGET_FPS, (OUT_SIZE, OUT_SIZE))
    for fr, c in zip(frames, cx):
        cropped = crop_one(fr, c, side, OUT_SIZE)
        writer.write(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))   # cv2 写 BGR
    writer.release()
    return len(frames), n_det


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--conf', type=float, default=0.5)
    ap.add_argument('--smooth_window', type=int, default=7)   # 7 帧=3.5s@2fps
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)

    videos = list_videos(args.input_dir)
    print(f'[adaptive_crop] {len(videos)} videos: {args.input_dir} -> {args.output_dir}')
    for i, vp in enumerate(videos):
        key = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(args.output_dir, key + '.mp4')
        if os.path.exists(out_path):
            print(f'  [{i+1}/{len(videos)}] skip (exists) {key}')
            continue
        try:
            nf, nd = process_video(vp, out_path, model, device, args.conf, args.smooth_window)
            print(f'  [{i+1}/{len(videos)}] {key}: {nf} frames @2fps, person detected in {nd}/{nf}')
        except Exception as e:
            print(f'  [{i+1}/{len(videos)}] ERROR {key}: {e}')
    print('[adaptive_crop] done.')
