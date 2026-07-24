import os, torch, json, tqdm, collections, random
from transformers import EvalPrediction

from .exo4d import Exo4D
from ..stream import StreamMixIn
from ..utils import ceil_time_by_fps, DictWithTo
from .pku import Pku
from .xdu import Xdu

class Exo4DClipStream(Exo4D, StreamMixIn):
    benchmarks_with_keys = {
        'goalstep': 'videos', 'fho_lta': 'clips', 'nlq': 'videos', 'moments': 'videos',
        'av': 'videos', 'fho_oscc-pnr': 'clips', 'fho_sta': 'annotations', 'vq': 'videos'
    }
    instructions = [{"role": "user", "content": "Please concisely narrate the video in real time. Use the tag 'C' to denote the camera wearer, and other letter tags, such as 'X', to denote other individuals in the scene."}]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')

    def get_annos(self, split: str) -> dict:
        anno_path = os.path.join(Exo4D.anno_root, f'{split}_clip.json')
        assert os.path.exists(anno_path)
        narration_streams = json.load(open(anno_path))
        return narration_streams

    def __init__(self, *, split: str, frame_fps: int, is_training: bool, augmentation: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, augmentation=augmentation, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps

        annos = self.get_annos(split)
        self.annos = []
        for video_uid, narrations in tqdm.tqdm(annos.items(), desc=f'narration_stream_{split}...'):
            if video_uid not in self.metadata:
                continue
            duration = self.metadata[video_uid]['duration']
            if not narrations:
                continue
            for narration in narrations:
                start_time = ceil_time_by_fps(narration['start_time'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(narration['end_time'], frame_fps, min_time=0, max_time=duration)
                time = ceil_time_by_fps(narration['timestamp'], frame_fps, min_time=0, max_time=duration)
                if end_time >= duration:
                    continue
                elif start_time > time or start_time >= end_time:
                    continue
                elif time > end_time:
                    continue 
                # text information
                history = narration['t1_caption']
                text = narration['text']
                flag = narration['flag']

                if flag == 0:
                    num_frames1 = int((time - start_time) * frame_fps + 1)
                    num_frames2 = int((end_time - time) * frame_fps)
                    if num_frames2 > 0:
                        conversation = [
                                {"role": "assistant", "content": history,  'learn': False},
                                {"role": "stream", 'num_frames': num_frames1, 'learn': True},
                                {"role": "assistant", "content": text, 'learn': True},
                                {"role": "stream", 'num_frames': num_frames2, 'learn': True}]
                    elif num_frames2 == 0:
                         conversation = [
                                {"role": "assistant", "content": history,  'learn': False},
                                {"role": "stream", 'num_frames': num_frames1, 'learn': True},
                                {"role": "assistant", "content": text, 'learn': True}]
                         end_time = time

                else:
                    num_frames = int((end_time - start_time) * frame_fps + 1)
                    conversation = [
                            {"role": "assistant", "content": history,  'learn': False},
                            {"role": "stream", 'num_frames': num_frames, 'learn': True}]

                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(int(start_time*frame_fps), int(end_time*frame_fps)+1)}
                })

    def __getitem__(self, index):
        anno = self.annos[index]
        return (*super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs)

    @staticmethod
    def _clean_text(src: str):
        # 1. remove #
        dst = src.replace('#C', '').replace('#c', '').replace('@c', '')
        dst = dst.replace('#O', '').replace('#o', '')
        dst = dst.replace('#Unsure', '').replace('#unsure', '')
        dst = dst.replace('#', '')
        # 2. remove start&end extra space and ,.
        dst = dst.strip('.,\n ') + '.'
        # 3. make the first word capitalize and remove extra space within the sentence
        words = dst.split()
        words[0] = words[0].capitalize()
        dst = ' '.join(words)
        return dst

    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        lm_ppl, frame_diff, fluency, lm_correctness = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
        return {
            f'lm_ppl': lm_ppl,
            f'time_diff': frame_diff / self.frame_fps,
            f'fluency': fluency,
            f'lm_correctness': lm_correctness
        }
        
class PkuClipStream(Pku, StreamMixIn):
    benchmarks_with_keys = {
        'goalstep': 'videos', 'fho_lta': 'clips', 'nlq': 'videos', 'moments': 'videos',
        'av': 'videos', 'fho_oscc-pnr': 'clips', 'fho_sta': 'annotations', 'vq': 'videos'
    }
    instructions = [{"role": "user", "content": "Please concisely narrate the video in real time. Use the tag 'C' to denote the camera wearer, and other letter tags, such as 'X', to denote other individuals in the scene."}]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')

    def get_annos(self, split: str) -> dict:
        anno_file = f'{split}.json'
        if split == 'train':
            anno_file = os.environ.get('PKU_TRAIN_JSON', anno_file)
            if not anno_file.endswith('.json'):
                anno_file = f'{anno_file}.json'
        anno_path = anno_file if os.path.isabs(anno_file) else os.path.join(Pku.anno_root, anno_file)
        assert os.path.exists(anno_path), f'PKUMMD annotation file not found: {anno_path}'
        print(f'[PkuClipStream] load annotations: {anno_path}')
        narration_streams = json.load(open(anno_path))
        return narration_streams

    def __init__(self, *, split: str, frame_fps: int, is_training: bool, augmentation: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, augmentation=augmentation, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps

        annos = self.get_annos(split)
        self.annos = []
        for video_uid, narrations in tqdm.tqdm(annos.items(), desc=f'narration_stream_{split}...'):
            if video_uid not in self.metadata:
                continue
            duration = self.metadata[video_uid]['duration']
            if not narrations:
                continue
            for narration in narrations:
                start_time = ceil_time_by_fps(narration['start_time'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(narration['end_time'], frame_fps, min_time=0, max_time=duration)
                time = ceil_time_by_fps(narration['timestamp'], frame_fps, min_time=0, max_time=duration)
                if end_time >= duration:
                    continue
                elif start_time > time or start_time >= end_time:
                    continue
                elif time > end_time:
                    continue 
                # text information
                history = narration['t1_caption']
                text = narration['text']
                flag = narration['flag']

                if flag == 0:
                    num_frames1 = int((time - start_time) * frame_fps + 1)
                    num_frames2 = int((end_time - time) * frame_fps)
                    if num_frames2 > 0:
                        conversation = [
                                {"role": "assistant", "content": history,  'learn': False},
                                {"role": "stream", 'num_frames': num_frames1, 'learn': True},
                                {"role": "assistant", "content": text, 'learn': True},
                                {"role": "stream", 'num_frames': num_frames2, 'learn': True}]
                    elif num_frames2 == 0:
                         conversation = [
                                {"role": "assistant", "content": history,  'learn': False},
                                {"role": "stream", 'num_frames': num_frames1, 'learn': True},
                                {"role": "assistant", "content": text, 'learn': True}]
                         end_time = time

                else:
                    num_frames = int((end_time - start_time) * frame_fps + 1)
                    conversation = [
                            {"role": "assistant", "content": history,  'learn': False},
                            {"role": "stream", 'num_frames': num_frames, 'learn': True}]

                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(int(start_time*frame_fps), int(end_time*frame_fps)+1)}
                })

    def __getitem__(self, index):
        anno = self.annos[index]
        return (*super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs)

    @staticmethod
    def _clean_text(src: str):
        # 1. remove #
        dst = src.replace('#C', '').replace('#c', '').replace('@c', '')
        dst = dst.replace('#O', '').replace('#o', '')
        dst = dst.replace('#Unsure', '').replace('#unsure', '')
        dst = dst.replace('#', '')
        # 2. remove start&end extra space and ,.
        dst = dst.strip('.,\n ') + '.'
        # 3. make the first word capitalize and remove extra space within the sentence
        words = dst.split()
        words[0] = words[0].capitalize()
        dst = ' '.join(words)
        return dst

    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        lm_ppl, frame_diff, fluency, lm_correctness = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
        return {
            f'lm_ppl': lm_ppl,
            f'time_diff': frame_diff / self.frame_fps,
            f'fluency': fluency,
            f'lm_correctness': lm_correctness
        }
        
class XduClipStream(Xdu, StreamMixIn):
    benchmarks_with_keys = {
        'goalstep': 'videos', 'fho_lta': 'clips', 'nlq': 'videos', 'moments': 'videos',
        'av': 'videos', 'fho_oscc-pnr': 'clips', 'fho_sta': 'annotations', 'vq': 'videos'
    }
    instructions = [{"role": "user", "content": "Please concisely narrate the video in real time. Use the tag 'C' to denote the camera wearer, and other letter tags, such as 'X', to denote other individuals in the scene."}]
    evaluation_kwargs = DictWithTo(evaluator='stream_evaluate')

    def get_annos(self, split: str) -> dict:
        anno_path = os.path.join(Xdu.anno_root, f'{split}.json')
        assert os.path.exists(anno_path)
        narration_streams = json.load(open(anno_path))
        return narration_streams

    def __init__(self, *, split: str, frame_fps: int, is_training: bool, augmentation: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, augmentation=augmentation, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps

        annos = self.get_annos(split)
        self.annos = []
        for video_uid, narrations in tqdm.tqdm(annos.items(), desc=f'narration_stream_{split}...'):
            if video_uid not in self.metadata:
                continue
            duration = self.metadata[video_uid]['duration']
            if not narrations:
                continue
            for narration in narrations:
                start_time = ceil_time_by_fps(narration['start_time'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(narration['end_time'], frame_fps, min_time=0, max_time=duration)
                time = ceil_time_by_fps(narration['timestamp'], frame_fps, min_time=0, max_time=duration)
                skip_time = ceil_time_by_fps(narration['skip_frame'], frame_fps, min_time=0, max_time=duration) if 'skip_frame' in narration else None
                
                if end_time >= duration:
                    continue
                elif start_time > time or start_time >= end_time:
                    continue
                elif time > end_time:
                    continue 
                if skip_time is not None:
                    if skip_time < start_time or skip_time >= time:
                        continue
                # text information
                history = narration['t1_caption']
                text = narration['text']
                flag = narration['flag']

                if flag == 0:
                    num_frames1 = int((time - start_time) * frame_fps + 1)
                    num_frames2 = int((end_time - time) * frame_fps)
                    num_frames_skip = int((skip_time - start_time) * frame_fps + 1) if skip_time is not None else None
                    if num_frames2 > 0:
                        if num_frames_skip is None:
                            conversation = [
                                    {"role": "assistant", "content": history,  'learn': False},
                                    {"role": "stream", 'num_frames': num_frames1, 'learn': True},
                                    {"role": "assistant", "content": text, 'learn': True},
                                    {"role": "stream", 'num_frames': num_frames2, 'learn': True}]
                        else:
                            conversation = [
                                    {"role": "assistant", "content": history,  'learn': False},
                                    {"role": "stream", 'num_frames': num_frames1, 'learn': num_frames_skip},
                                    {"role": "assistant", "content": text, 'learn': True},
                                    {"role": "stream", 'num_frames': num_frames2, 'learn': True}]
                    elif num_frames2 == 0:
                        if num_frames_skip is None:
                            conversation = [
                                    {"role": "assistant", "content": history,  'learn': False},
                                    {"role": "stream", 'num_frames': num_frames1, 'learn': True},
                                    {"role": "assistant", "content": text, 'learn': True}]
                            end_time = time
                        else:
                            conversation = [
                                    {"role": "assistant", "content": history,  'learn': False},
                                    {"role": "stream", 'num_frames': num_frames1, 'learn': num_frames_skip},
                                    {"role": "assistant", "content": text, 'learn': True}]
                            end_time = time

                else:
                    num_frames = int((end_time - start_time) * frame_fps + 1)
                    conversation = [
                            {"role": "assistant", "content": history,  'learn': False},
                            {"role": "stream", 'num_frames': num_frames, 'learn': True}]

                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(int(start_time*frame_fps), int(end_time*frame_fps)+1)}
                })

    def __getitem__(self, index):
        anno = self.annos[index]
        return (*super().__getitem__(
            conversation=anno['conversation'],
            load_ranges=anno['load_ranges'],
        ), index, self.evaluation_kwargs)

    @staticmethod
    def _clean_text(src: str):
        # 1. remove #
        dst = src.replace('#C', '').replace('#c', '').replace('@c', '')
        dst = dst.replace('#O', '').replace('#o', '')
        dst = dst.replace('#Unsure', '').replace('#unsure', '')
        dst = dst.replace('#', '')
        # 2. remove start&end extra space and ,.
        dst = dst.strip('.,\n ') + '.'
        # 3. make the first word capitalize and remove extra space within the sentence
        words = dst.split()
        words[0] = words[0].capitalize()
        dst = ' '.join(words)
        return dst

    def compute_metrics(self, eval_predictions: EvalPrediction, *args, **kwargs):
        lm_ppl, frame_diff, fluency, lm_correctness = torch.from_numpy(eval_predictions.predictions).mean(dim=0).tolist()
        return {
            f'lm_ppl': lm_ppl,
            f'time_diff': frame_diff / self.frame_fps,
            f'fluency': fluency,
            f'lm_correctness': lm_correctness
        }

def build_exo4d_clip_stream_train(**kwargs):
    return Exo4DClipStream(split='train', **kwargs)

def build_exo4d_clip_stream_val(**kwargs):
    return Exo4DClipStream(split='val', **kwargs)

def build_pku_clip_stream_train(**kwargs):
    return PkuClipStream(split='train', **kwargs)

def build_pku_clip_stream_val(**kwargs):
    return PkuClipStream(split='val', **kwargs)

def build_xdu_clip_stream_train(**kwargs):
    return XduClipStream(split='train', **kwargs)

def build_xdu_clip_stream_val(**kwargs):
    return XduClipStream(split='val', **kwargs)

class XduNoSilentClipStream(XduClipStream):
    # EOS silent loss ablation: keep the same visual/text samples, but remove
    # all learnable stream labels. Silent-only samples are dropped to avoid
    # zero-label batches after stream supervision is removed.
    @staticmethod
    def _shrink_load_ranges(load_ranges: dict[str, range], num_frames: int):
        remaining = num_frames
        new_load_ranges = {}
        for path, ranger in load_ranges.items():
            take = min(remaining, len(ranger))
            if take > 0:
                new_load_ranges[path] = range(ranger.start, ranger.start + take)
            remaining -= take
            if remaining <= 0:
                break
        return new_load_ranges

    def __init__(self, *, split: str, **kwargs):
        super().__init__(split=split, **kwargs)
        kept_annos = []
        dropped = 0
        for anno in self.annos:
            for message in anno['conversation']:
                if message['role'] == 'stream':
                    message['learn'] = False
            while anno['conversation'] and anno['conversation'][-1]['role'] == 'stream':
                anno['conversation'].pop()
            num_frames = sum(
                message['num_frames'] for message in anno['conversation']
                if message['role'] == 'stream'
            )
            anno['load_ranges'] = self._shrink_load_ranges(anno['load_ranges'], num_frames)
            clipped_conversation, _ = self.max_frames_clip(
                anno['conversation'], anno['load_ranges'], self.max_num_frames)
            has_text_label = any(
                message['role'] == 'assistant' and message.get('learn', False)
                for message in clipped_conversation
            )
            if has_text_label:
                kept_annos.append(anno)
            else:
                dropped += 1
        print(f'[XduNoSilentClipStream] drop {dropped} no-label samples after removing stream labels')
        self.annos = kept_annos

def build_xdu_no_silent_clip_stream_train(**kwargs):
    return XduNoSilentClipStream(split='train', **kwargs)

def build_xdu_no_silent_clip_stream_val(**kwargs):
    return XduNoSilentClipStream(split='val', **kwargs)

class XduCropClipStream(XduClipStream):
    # ASTime 自适应裁剪 384 版本(用于 PKUMMD+ASTime 联合训练)。
    # 仅覆盖 root: 特征读 dataset/ASTime_crop384/features、metadata 也在该目录;
    # 标注仍复用 dataset/ASTime/annotations(裁剪只改空间、不改时间, get_annos 用 Xdu.anno_root)。
    root = 'dataset/ASTime_crop384'

def build_xdu_crop_clip_stream_train(**kwargs):
    return XduCropClipStream(split='train', **kwargs)

def build_xdu_crop_clip_stream_val(**kwargs):
    return XduCropClipStream(split='val', **kwargs)

# ===== FPS 敏感性实验:每个帧率一套独立特征/metadata(root 隔离),标注复用 ASTime(降帧不改时间) =====
# 训练时务必配套传 --frame_fps {F} 与 --max_num_frames {120*F}(保持 120s 覆盖窗口恒定)。
class XduFps1ClipStream(XduClipStream):
    root = 'dataset/ASTime_fps1'

class XduFps4ClipStream(XduClipStream):
    root = 'dataset/ASTime_fps4'

class XduFps8ClipStream(XduClipStream):
    root = 'dataset/ASTime_fps8'

def build_xdu_fps1_clip_stream_train(**kwargs):
    return XduFps1ClipStream(split='train', **kwargs)

def build_xdu_fps4_clip_stream_train(**kwargs):
    return XduFps4ClipStream(split='train', **kwargs)

def build_xdu_fps8_clip_stream_train(**kwargs):
    return XduFps8ClipStream(split='train', **kwargs)

if __name__ == '__main__':
    build_exo4d_clip_stream_train(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )
    build_exo4d_clip_stream_val(
        frame_fps=2, is_training=True, augmentation=True,
        system_prompt='', tokenizer=None,
        vision_pretrained='google/siglip-large-patch16-384',
        embed_mark='2fps_384_1+3x3'
    )
