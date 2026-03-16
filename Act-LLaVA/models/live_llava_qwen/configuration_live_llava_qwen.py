
from transformers import Qwen2Config, AutoConfig

from ..configuration_live import LiveConfigMixin
from llava.model import *

class LiveLlavaQwenConfig(LlavaQwenConfig, LiveConfigMixin):
    pass