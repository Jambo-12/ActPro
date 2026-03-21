import os
import sys
import torchvision
import transformers
import tqdm
import time
import json
import argparse
import torch.multiprocessing as mp

# torchvision.set_video_backend('video_reader')
torchvision.set_video_backend('pyav')

from data.utils import ffmpeg_once, list_videos
from .inference import LiveInferClip

logger = transformers.logging.get_logger('liveinfer')

def main(liveinfer: LiveInferClip, args):
    data_path = args.data_path
    sam_path = args.sam_path
    save_path = args.save_path
    dataset_type = args.dataset_type

    for src_video_path in list_videos(data_path):
        try:
            liveinfer.reset()
            rel_path = os.path.relpath(src_video_path, start=data_path)
            name, ext = os.path.splitext(rel_path)

            if dataset_type == 'pkummd':
                try:
                    video_id = int(name.split('/')[-1].split('-')[0])
                    if video_id < 291 or video_id > 334: # 291-334 are the test set videos for PKUMMD
                        continue
                except ValueError:
                    logger.warning(f"Could not parse video_id from {name}. Skipping filter.")

            ffmpeg_video_path = os.path.join(sam_path + f'_{liveinfer.frame_fps}fps',  rel_path)

            if not os.path.exists(os.path.dirname(ffmpeg_video_path)):
                os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                
            save_history_path = os.path.join(save_path, name + '.json')
            if not os.path.exists(os.path.dirname(save_history_path)):
                os.makedirs(os.path.dirname(save_history_path), exist_ok=True)
            
            # 检查 save_history_path 文件是否存在
            if os.path.exists(save_history_path):
                logger.info(f"History for {src_video_path} already exists. Skipping...")
                continue
            
            if not os.path.exists(ffmpeg_video_path):
                os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
                ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
                logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
            
            liveinfer.load_video(ffmpeg_video_path)
            first_assistant = "The person is standing."
            liveinfer.input_memory_stream(first_assistant, video_time=0)

            timecosts = []
            pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
            history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
            
            for i in range(liveinfer.num_video_frames):
                start_time = time.time()
                liveinfer.input_video_stream(i / liveinfer.frame_fps)
                query, response = liveinfer()
                end_time = time.time()
                
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
            
            with open(save_history_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f'The conversation history has been saved to {save_history_path}.')
        
        except Exception as e:
            logger.error(f"An error occurred while processing {src_video_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiveInfer processing for multiple datasets")
    parser.add_argument('--data_path', type=str, required=True, help="Path to source videos")
    parser.add_argument('--sam_path', type=str, required=True, help="Path to save SAM/ffmpeg videos")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save result JSONs")
    parser.add_argument('--dataset_type', type=str, choices=['ASTime', 'PKUMMD'], default='ASTime', 
                        help="Choose dataset type to apply specific formatting/filtering rules")
    
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown
    liveinfer = LiveInferClip()
    main(liveinfer, args)