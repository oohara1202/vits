import os
import argparse
import torch

import commons
from text import text_to_sequence

def get_parser():
    parser = argparse.ArgumentParser(description='Inference evalation data.')
    parser.add_argument(
        'model_dir',
        type=str,
        help='Path of model directory',
    )
    return parser

# 与えられたモデルのconfig.jsonを返す
def get_config(model_path):
    config_file = os.path.join(os.path.dirname(model_path), 'config.json')
    assert os.path.exists(config_file)
    return config_file

# テキストを所定のsequenceへ変換
def get_normed_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
