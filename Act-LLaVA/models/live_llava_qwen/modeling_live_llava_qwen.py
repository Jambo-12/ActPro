import torch
from torch import nn
from transformers import Cache, Qwen2Model, Qwen2ForCausalLM, Qwen2Config
from transformers.activations import GELUActivation
from transformers.utils import logging

from .configuration_live_llava_qwen import LiveLlavaQwenConfig
from ..modeling_live import build_live, LiveMixin
from llava.model import *
from llava.model.llava_arch import *

logger = logging.get_logger(__name__)

class LiveLlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LiveLlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LiveLlavaQwenModel, self).__init__(config)

class LiveLlavaQwenForCausalLM(Qwen2ForCausalLM, LiveMixin):
    config_class = LiveLlavaQwenConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlavaQwenConfig):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "live_llava_qwen"
        config.rope_scaling = None

        self.model = LiveLlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        frames: torch.FloatTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        cache_position: torch.LongTensor = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().forward(
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            # labels
            use_cache = use_cache,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict)

        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()
            loss_text = (loss * ~v_mask).sum() / (labels >= 0).sum()
            # print(loss_text)

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs
    
        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)

def build_live_llava_qwen(**kwargs):
    return build_live(config_class=LiveLlavaQwenConfig, model_class=LiveLlavaQwenForCausalLM, **kwargs)

if __name__ == '__main__':
    from ..arguments_live import LiveOnePlusTrainingArguments
    print(LiveOnePlusTrainingArguments().to_dict())
    model, tokenizer = build_live_llava_qwen(is_training=True, **LiveOnePlusTrainingArguments().to_dict())
    print(model.config, tokenizer)
