import torch
from torchvision.io import read_video

from .inference import LiveInferClip, logger


class LiveInferClipCPUVideo(LiveInferClip):
    """Keep decoded video frames on CPU and move only requested frames to CUDA."""

    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        if frame_idx > self.last_frame_idx:
            ranger = range(self.last_frame_idx + 1, frame_idx + 1)
            frames = self.video_tensor[ranger].to('cuda', non_blocking=True)
            frames_embeds = self.model.visual_embed(frames).split(self.frame_num_tokens)
            del frames
            self.frame_embeds_queue.extend([
                (r / self.frame_fps, frame_embeds)
                for r, frame_embeds in zip(ranger, frames_embeds)
            ])
        self.last_frame_idx = frame_idx
        self.video_time = video_time

    def load_video(self, video_path):
        self.video_tensor = read_video(video_path, pts_unit='sec', output_format='TCHW')[0]
        self.num_video_frames = self.video_tensor.size(0)
        self.video_duration = self.video_tensor.size(0) / self.frame_fps
        logger.warning(f'{video_path} -> {self.video_tensor.shape}, {self.frame_fps} FPS, CPU video tensor')
