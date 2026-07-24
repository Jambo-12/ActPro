import argparse
import json
import os
import statistics
import sys
import time

import torch
import torchvision
import tqdm
import transformers

torchvision.set_video_backend('pyav')

from data.utils import ffmpeg_once, list_videos
from .inference import LiveInferClip
from .inference_cpu_video import LiveInferClipCPUVideo

logger = transformers.logging.get_logger('liveinfer')


def _gb(num_bytes):
    return round(num_bytes / (1024 ** 3), 3)


def _cuda_stats():
    if not torch.cuda.is_available():
        return {}
    return {
        'allocated_gb': _gb(torch.cuda.memory_allocated()),
        'reserved_gb': _gb(torch.cuda.memory_reserved()),
        'max_allocated_gb': _gb(torch.cuda.max_memory_allocated()),
        'max_reserved_gb': _gb(torch.cuda.max_memory_reserved()),
    }


def _summarize(per_video, failed, skipped, total_wall_clock_sec):
    completed = [item for item in per_video if item['status'] == 'completed']
    total_frames = sum(item['num_frames'] for item in completed)
    frame_loop_sec = sum(item['frame_loop_wall_clock_sec'] for item in completed)
    per_video_fps = [item['final_processing_fps'] for item in completed]
    summary = {
        'num_completed': len(completed),
        'num_failed': len(failed),
        'num_skipped': len(skipped),
        'total_frames': total_frames,
        'total_wall_clock_sec': round(total_wall_clock_sec, 3),
        'frame_loop_wall_clock_sec': round(frame_loop_sec, 3),
        'weighted_processing_fps': round(total_frames / frame_loop_sec, 4) if frame_loop_sec else None,
        'macro_avg_processing_fps': round(statistics.mean(per_video_fps), 4) if per_video_fps else None,
        'median_processing_fps': round(statistics.median(per_video_fps), 4) if per_video_fps else None,
        'failed_videos': failed,
        'skipped_videos': skipped,
        'per_video': per_video,
    }
    if completed:
        summary['peak_allocated_gb'] = max(item.get('max_allocated_gb', 0) for item in completed)
        summary['peak_reserved_gb'] = max(item.get('max_reserved_gb', 0) for item in completed)
    return summary


def main(liveinfer, args):
    data_path = args.data_path
    sam_path = args.sam_path
    save_path = args.save_path
    dataset_type_lc = args.dataset_type.lower()

    os.makedirs(save_path, exist_ok=True)
    if args.timing_summary:
        os.makedirs(os.path.dirname(args.timing_summary), exist_ok=True)

    if torch.cuda.is_available():
        model_loaded_stats = _cuda_stats()
        torch.cuda.reset_peak_memory_stats()
    else:
        model_loaded_stats = {}

    per_video = []
    failed = []
    skipped = []
    run_start = time.perf_counter()

    for src_video_path in list_videos(data_path):
        name = None
        try:
            liveinfer.reset()
            rel_path = os.path.relpath(src_video_path, start=data_path)
            name, ext = os.path.splitext(rel_path)

            if dataset_type_lc == 'pkummd':
                try:
                    video_id = int(name.split('/')[-1].split('-')[0])
                    if video_id < 291 or video_id > 334:
                        continue
                except ValueError:
                    logger.warning(f"Could not parse video_id from {name}. Skipping filter.")

            ffmpeg_video_path = os.path.join(sam_path + f'_{liveinfer.frame_fps}fps', rel_path)
            save_history_path = os.path.join(save_path, name + '.json')
            os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
            os.makedirs(os.path.dirname(save_history_path), exist_ok=True)

            if os.path.exists(save_history_path):
                logger.info(f"History for {src_video_path} already exists. Skipping...")
                skipped.append(name)
                continue

            if not os.path.exists(ffmpeg_video_path):
                ffmpeg_resolution = None if dataset_type_lc == 'astime' else liveinfer.frame_resolution
                ffmpeg_crop_center = dataset_type_lc == 'pkummd'
                ffmpeg_once(
                    src_video_path,
                    ffmpeg_video_path,
                    fps=liveinfer.frame_fps,
                    resolution=ffmpeg_resolution,
                    crop_center=ffmpeg_crop_center,
                )
                logger.warning(
                    f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, '
                    f'resolution={ffmpeg_resolution}, crop_center={ffmpeg_crop_center}'
                )

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            video_start = time.perf_counter()
            liveinfer.load_video(ffmpeg_video_path)
            liveinfer.input_memory_stream("The person is standing.", video_time=0)

            timecosts = []
            history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []}
            pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")

            frame_loop_start = time.perf_counter()
            for i in range(liveinfer.num_video_frames):
                start_time = time.perf_counter()
                liveinfer.input_video_stream(i / liveinfer.frame_fps)
                query, response = liveinfer()
                end_time = time.perf_counter()

                timecosts.append(end_time - start_time)
                fps = (i + 1) / sum(timecosts)
                pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
                pbar.update(1)

                if query:
                    history['conversation'].append({'role': 'user', 'content': query, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
                    print(query)
                if response:
                    history['conversation'].append({'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})
                    print(response)
                if not query and not response:
                    history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})

            frame_loop_sec = time.perf_counter() - frame_loop_start
            video_wall_sec = time.perf_counter() - video_start

            with open(save_history_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f'The conversation history has been saved to {save_history_path}.')

            video_stats = {
                'name': name,
                'status': 'completed',
                'num_frames': liveinfer.num_video_frames,
                'video_duration_sec': round(liveinfer.num_video_frames / liveinfer.frame_fps, 3),
                'wall_clock_sec': round(video_wall_sec, 3),
                'frame_loop_wall_clock_sec': round(frame_loop_sec, 3),
                'final_processing_fps': round(liveinfer.num_video_frames / sum(timecosts), 4),
                'mean_step_cost_sec': round(statistics.mean(timecosts), 6),
            }
            video_stats.update(_cuda_stats())
            per_video.append(video_stats)
            torch.cuda.empty_cache()

        except Exception as e:
            failed_name = name or src_video_path
            failed.append(failed_name)
            logger.error(f"An error occurred while processing {src_video_path}: {e}")
            torch.cuda.empty_cache()

    total_wall_clock_sec = time.perf_counter() - run_start
    summary = _summarize(per_video, failed, skipped, total_wall_clock_sec)
    summary.update({
        'frame_fps': liveinfer.frame_fps,
        'save_path': save_path,
        'sam_path': sam_path,
        'data_path': data_path,
        'cpu_video': args.cpu_video,
        'model_loaded_cuda': model_loaded_stats,
    })

    print('===== TIMING SUMMARY =====')
    print(json.dumps(summary, indent=2))
    if args.timing_summary:
        with open(args.timing_summary, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'Timing summary saved to {args.timing_summary}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiveInfer timing for multiple datasets")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sam_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, choices=['ASTime', 'PKUMMD'], default='ASTime')
    parser.add_argument('--timing_summary', type=str, default=None)
    parser.add_argument('--cpu_video', action='store_true')

    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    infer_cls = LiveInferClipCPUVideo if args.cpu_video else LiveInferClip
    liveinfer = infer_cls()
    main(liveinfer, args)
