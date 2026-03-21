import math, torch
from functools import partial
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from llava.mm_utils import process_images, image_token_generation
from .configuration_live import LiveConfigMixin
import torch.nn.functional as F

def _siglip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    frames = F.interpolate(frames, size=(384, 384), mode='bilinear', align_corners=False)
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state.reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = vision_outputs.pooler_output[:, None]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def _clip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state[:,1:].reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = last_hidden_state[:,0]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def _llava_siglip_vision_encode(vision_model, frames, config, **kwargs):

    image_processor = vision_model.image_processor
    device = vision_model.device
    image_tensor = process_images(frames, image_processor, config)

    # 将预处理后的图片张量（image_tensor）转换为 float16 数据类型，并传输到计算设备 (GPU) 上。
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    # 对图片张量进行进一步处理，调用模型生成对应的图像 token。
    # image_token_generation 是外部定义的函数，负责对输入的图像进行特征提取（如嵌入向量）。
    # 批量大小设为了 16，保证在计算资源下不会溢出。
    image_tokens = image_token_generation(vision_tower=vision_model, images=image_tensor, batch_size=16)

    # 返回最后生成的图像 tokens，可作为后续任务（如文本生成、分类等）的输入。
    return image_tokens



def build_live_vision(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.vision_pretrained).vision_model
    if 'google/siglip-large-patch16-384' in config.vision_pretrained:
        return model, partial(_siglip_vision_encode, frame_token_cls=config.frame_token_cls, frame_token_pooled=config.frame_token_pooled)
    elif 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90k' == config.vision_pretrained or 'openai/clip-vit-large-patch14-336' == config.vision_pretrained:
        return model, partial(_clip_vision_encode, config)
    else:
        raise ValueError(f'Unverified vision_pretrained: {config.vision_pretrained}')