
import os
import json
import math
import argparse
import datetime
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.io.wavfile import write

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from abelab_utils.g2p import G2P

def main():
    text = 'あらゆる現実をすべて自分の方へ捻じ曲げたのだ'

    now = str(datetime.datetime.now())
    day, time = now.split('.')[0].split(' ')
    dpath = os.path.join(
        'generated_wav',
        day.replace('-', '') + '_' + time.replace(':', '')
    )

    os.makedirs(dpath, exist_ok=True)

    args = get_parser().parse_args()

    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None)

    stn_tst = get_phonetic_prosodic_symbols(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    save_path = os.path.join(dpath, f'{text}.wav')
    write(save_path, hps.data.sampling_rate, audio)

def get_parser():
    parser = argparse.ArgumentParser(description='Inference on VITS (Ja).')
    parser.add_argument(
        'model',
        type=str,
        help='Path of model file',
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path of config file',
    )
    return parser

def get_phonetic_prosodic_symbols(text, hps):
    text_norm = text_to_sequence(' '.join(G2P.from_grapheme(text)), hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)

    return text_norm

if __name__ == "__main__":
  main()
