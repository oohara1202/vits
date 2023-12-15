import os
import argparse

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
