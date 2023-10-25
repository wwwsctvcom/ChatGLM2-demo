import random
import torch
import numpy as np
from transformers import set_seed
from datetime import datetime


def seed_everything(seed: int = 42) -> None:
    if seed:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_cur_time() -> str:
    """
    return: 1970-01-01
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_cur_time_sec() -> str:
    """
    return: 1970-01-01 00:00:00
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def build_inputs(query: str = None, history=None):
    if history is None:
        history = []
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    # 添加->用于区分最后一轮和history
    prompt += "[Round {}]\n\n问：{} -> \n\n答：".format(len(history) + 1, query)
    return prompt
