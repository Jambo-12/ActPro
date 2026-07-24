from transformers import HfArgumentParser

from .arguments_live import LiveTrainingArguments, get_args_class
from .live_llava_qwen import build_live_llava_qwen as build_model_and_tokenizer
from .modeling_live import fast_greedy_generate

def parse_args() -> LiveTrainingArguments:
    # 第一次解析只为读出 live_version,因此容忍只在具体子类(如 live1+ 的 max_num_frames)里才有的参数,
    # 不在此处对未知参数报错;真正的严格校验交给第二次按具体类的解析(下行,默认仍会拒绝未知参数)。
    args, *_ = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    args, = HfArgumentParser(get_args_class(args.live_version)).parse_args_into_dataclasses()
    return args