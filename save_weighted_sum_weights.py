# python save_weighted_sum_weights.py logs/studies-teacher_sslfeature_ft
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models import SynthesizerTrn
from text.symbols import symbols

import inference_utils

SAVE_PATH = 'dump/averaged_vector_studies-teacher/weighted_sum_weights.pt'

def main():
    args = inference_utils.get_parser().parse_args()
    model_dir = args.model_dir
    assert os.path.isdir(model_dir)

    # latestモデルのフルパス
    print('Model: ', end='')
    model_path = utils.latest_checkpoint_path(model_dir, 'G_*.pth')
    
    hps = utils.get_hparams_from_file(inference_utils.get_config(model_path))

    # Referenceのembeddingを使うかどうか
    try:
        use_embed = hps.data.use_embed
        try:
            use_embed_ssl = hps.data.use_embed_ssl
        except:
            use_embed_ssl = False
        embed_dim = hps.data.embed_dim
    except:
        use_embed = False
        use_embed_ssl = False
        embed_dim = None

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers = hps.data.n_speakers,
        use_embed = use_embed,          # for embedding conditioning
        use_embed_ssl = use_embed_ssl,  # for embedding conditioning
        embed_dim = embed_dim,          # for embedding conditioning
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)

    weights = net_g.state_dict()['ulf.weights'].cpu()
    print('Weights: ')
    print(weights)
    norm_weights = F.softmax(weights, dim=-1)
    print('Norm Weights: ')
    print(norm_weights)

    torch.save(weights, SAVE_PATH)
            
if __name__ == "__main__":
    main()
