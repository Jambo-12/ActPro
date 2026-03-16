import os, torchvision, transformers, tqdm, time, json
import torch.multiprocessing as mp
# torchvision.set_video_backend('video_reader')
torchvision.set_video_backend('pyav')

from data.utils import ffmpeg_once, list_videos

from .inference import LiveInferClip
logger = transformers.logging.get_logger('liveinfer')

# python -m demo.cli --resume_from_checkpoint ... 

def main(liveinfer: LiveInferClip):
    # data_path = '/rest/jambo/Datasets/timestampData/TMM_revision'
    # sam_path = '/work/pqz/dataset/TMM_version/sam_videos'
    # save_path = '/work/pqz/dataset/TMM_version/results/pku_jsonv8_all'
    data_path = '/rest/jambo/Datasets/PKUMMD/TMM_revision'
    sam_path = '/work/pqz/dataset/TMM_version/sam_videos'
    save_path = '/work/pqz/dataset/TMM_version/results_test/pku_jsonv8_all_v2'
    # file_info = json.load(open(json_path, 'r'))
    for src_video_path in list_videos(data_path):
        try:
            liveinfer.reset()
            rel_path = os.path.relpath(src_video_path, start=data_path)
            name, ext = os.path.splitext(rel_path)
            # create folder for saving
            ffmpeg_video_path = os.path.join(sam_path, rel_path)
            # ffmpeg_video_path = src_video_path
            if not os.path.exists(os.path.dirname(ffmpeg_video_path)):
                os.makedirs(os.path.dirname(ffmpeg_video_path))
            save_history_path = os.path.join(save_path, name + '.json')
            if not os.path.exists(os.path.dirname(save_history_path)):
                os.makedirs(os.path.dirname(save_history_path))
            
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
            # liveinfer.input_query_stream('Please narrate the video in real time.', video_time=0.0)
            # liveinfer.input_query_stream('Hi, who are you?', video_time=1.0)
            # liveinfer.input_query_stream('Yes, I want to check its safety.', video_time=3.0)   
            # liveinfer.input_query_stream('No, I am going to install something to alert pedestrians to move aside. Could you guess what it is?', video_time=12.5)

            timecosts = []
            pbar = tqdm.tqdm(total=liveinfer.num_video_frames, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
            history = {'video_path': src_video_path, 'frame_fps': liveinfer.frame_fps, 'conversation': []} 
            for i in range(liveinfer.num_video_frames):
                # liveinfer.frame_token_interval_threshold -= 0.00175 # decay
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
            json.dump(history, open(save_history_path, 'w'), indent=4)
            print(f'The conversation history has been saved to {save_history_path}.')
        
        except Exception as e:
            logger.error(f"An error occurred while processing {src_video_path}: {e}")
if __name__ == '__main__':
    liveinfer = LiveInferClip()
    main(liveinfer)