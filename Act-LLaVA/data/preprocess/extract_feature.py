import argparse
import functools
import os
import torch
import torchvision
import tqdm
import random
import submitit

from dataclasses import dataclass
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, image_token_generation

def distributed_encode(*, src_root: str,
                       model_config: dict,
                       vision_process: callable,
                       batch_size: int,
                       save_bf16: bool = False,
                       **kwargs):
    
    env = submitit.JobEnvironment()

    src_root = src_root.rstrip('/')
    model_name = model_config["model_name"]
    vision_pretrained = model_config["pretrained"]
    dst_root = model_config["target_root"].rstrip('/')

    device = f'cuda:{env.local_rank}'
    _, model, image_processor, _ = load_pretrained_model(vision_pretrained, None, model_name, device_map="auto") 

    model.to(device).eval()

    os.makedirs(dst_root, exist_ok=True)
    path_list = os.listdir(src_root)
    random.shuffle(path_list)
    
    for i, file in tqdm.tqdm(enumerate(path_list), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue 

        video_path = os.path.join(src_root, file)
        save_path = os.path.splitext(video_path)[0] + ".pt"
        save_path = save_path.replace(src_root, dst_root)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            continue

        try:
            frames = torchvision.io.read_video(video_path, pts_unit="sec", output_format="TCHW")[0]
            setattr(model.config, "image_aspect_ratio", "full")

            frames_tensor = vision_process(frames, image_processor, model.config)
            frames_tensor = [frame.to(dtype=torch.float16, device=device) for frame in frames_tensor]
            vision_tower = model.get_vision_tower()
            
            with torch.no_grad():
                features = image_token_generation(vision_tower=vision_tower, images=frames_tensor, batch_size=16)

            if save_bf16:
                features = features.to(dtype=torch.bfloat16)
            
            features = features.to(device='cpu')
            torch.save(features, save_path)

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Video Feature Extraction")
    
    parser.add_argument("--video_dir", type=str, default='dataset/PKUMMD/Videos_2fps', help="Path to the source video directory")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of compute nodes")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--pretrained", type=str, default="huggingface/lmms-lab/llava-onevision-qwen2-7b-ov", help="Path to the pretrained model")
    parser.add_argument("--slurm_partition", type=str, default=None, help="SLURM partition name if applicable")

    args = parser.parse_args()

    clean_video_dir = args.video_dir.rstrip('/')
    parent_dir = os.path.dirname(clean_video_dir)
    auto_target_root = os.path.join(parent_dir, "Features")

    print(f"[*] Source video directory: {clean_video_dir}")
    print(f"[*] Auto-generated features directory: {auto_target_root}")

    model_name = "llava_qwen"

    task_config = {
        "model_name": model_name,
        "pretrained": args.pretrained,
        "target_root": auto_target_root,
    }

    task_func = functools.partial(
        distributed_encode,
        src_root=clean_video_dir,
        model_config=task_config,
        vision_process=process_images,
        batch_size=1,
        save_bf16=True  
    )

    executor = submitit.AutoExecutor(folder="outputs/", cluster='local' if args.num_nodes == 1 else 'slurm')
    
    executor.update_parameters(
        tasks_per_node=args.num_gpus,
        nodes=args.num_nodes,
        gpus_per_node=args.num_gpus,
        cpus_per_task=1,
        slurm_partition=args.slurm_partition if args.slurm_partition else None,
        mem_gb=240,
        slurm_time='24:00:00',
        timeout_min=600,
    )

    job = executor.submit(task_func)
    print(f"[*] Job submitted! Job ID: {job.job_id}")
    job.results()