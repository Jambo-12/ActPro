import os, torchvision, transformers, tqdm, time, json
import argparse
import torch.multiprocessing as mp
torchvision.set_video_backend('pyav')
import sys

from data.utils import ffmpeg_once
from .inference import LiveInferClip

logger = transformers.logging.get_logger('liveinfer')

def main(liveinfer: LiveInferClip, video_path: str):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    src_video_path = video_path
    name, ext = os.path.splitext(src_video_path)
    
    ffmpeg_video_path = os.path.join(name + f'_{liveinfer.frame_fps}fps_{liveinfer.frame_resolution}' + ext)
    save_history_path = os.path.splitext(ffmpeg_video_path)[0] + '.json'
    
    if not os.path.exists(ffmpeg_video_path):
        os.makedirs(os.path.dirname(ffmpeg_video_path), exist_ok=True)
        ffmpeg_once(src_video_path, ffmpeg_video_path, fps=liveinfer.frame_fps, resolution=liveinfer.frame_resolution)
        logger.warning(f'{src_video_path} -> {ffmpeg_video_path}, {liveinfer.frame_fps} FPS, {liveinfer.frame_resolution} Resolution')
    
    liveinfer.load_video(ffmpeg_video_path)
    
    first_assistant = "The person stands on the floor."
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
        pbar.set_postfix_str(f"Avg FPS: {fps:.1f}")
        pbar.update(1)
        
        entry = {'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]}
        if query:
            entry['role'] = 'user'
            entry['content'] = query
            print(f"\nUser: {query}")
        if response:
            entry['role'] = 'assistant'
            entry['content'] = response
            print(f"\nAssistant: {response}")
        
        history['conversation'].append(entry)

    with open(save_history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f'\nConversation history saved to: {save_history_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiveInferClip CLI")
    
    # 1. Define only the arguments specific to this script
    parser.add_argument('--video_path', type=str, required=True, help='Path to video')
    
    # 2. Use parse_known_args to separate your args from the model's args
    args, remaining_args = parser.parse_known_args()

    # 3. Pass remaining_args to LiveInferClip if it handles sys.argv internally, 
    sys.argv = [sys.argv[0]] + remaining_args
    liveinfer = LiveInferClip() 
    
    main(liveinfer, args.video_path)