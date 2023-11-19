# python inference_eval_data.py logs/jsut_base
import os
import argparse
import scipy
import torch
import pickle

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

def main():
    args = _get_parser().parse_args()
    model_dir = args.model_dir

    # latestモデルのフルパス
    print('Model: ', end='')
    model_path = utils.latest_checkpoint_path(model_dir, 'G_*.pth')
    
    hps = utils.get_hparams_from_file(_get_config(model_path))

    # testデータのfilelistのpathを取得
    test_files = hps.data.training_files.replace('_train_', '_test_')
    assert os.path.isfile(test_files)

    # wavファイルの保存directory作成
    d = os.path.basename(model_dir)
    assert len(d) != 0  # logs/jsut_base/ のようにスラッシュで終わっていると怒る
    dname = os.path.join('wav', d)
    os.makedirs(dname, exist_ok=True)

    # multi-speakerかどうか
    if '_sid_' in test_files:
        multi_spk = True
    else:
        multi_spk = False

    # 話者分directory作成
    if multi_spk:
        for i in range(hps.data.n_speakers):
            for t in ['gt', 'gen']:  # ground truthと合成音声を比較する用
                os.makedirs(os.path.join(dname, f'spk{int(i)}', t), exist_ok=True)
    else:
        for t in ['gt', 'gen']:  # ground truthと合成音声を比較する用
            os.makedirs(os.path.join(dname, 'spk0', t), exist_ok=True)
 
    # Referenceのembeddingを使うかどうか
    try:
        use_embed = hps.data.use_embed
        embed_dim = hps.data.embed_dim
    except:
        use_embed = None
        embed_dim = None

    # embeddingベクトルの読み込み
    if use_embed:
        embed_dir = hps.data.embed_dir
        embed_dict = _get_embed_dict(embed_dir, test_files)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers = hps.data.n_speakers,
        use_embed = use_embed,  # for embedding conditioning
        embed_dim = embed_dim,  # for embedding conditioning
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)

    with open(test_files, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.rstrip()
        line_splitted = line.split('|')
        
        fpath = line_splitted[0]         # ファイルパス
        fname = os.path.basename(fpath)  # ファイル名
        text = line_splitted[-1]         # テキスト
        text_norm = _get_normed_text(text, hps)

        sid = None                       # 話者ID
        if multi_spk:                    #
            sid = line_splitted[1]       #
            save_dir = os.path.join(dname, f'spk{sid}', 'gen')  
            dst_path = os.path.join(dname, f'spk{sid}', 'gt', fname)
            sid = torch.LongTensor([int(sid)]).cuda()
        else:                            # 
            save_dir = os.path.join(dname, 'spk0', 'gen')
            dst_path = os.path.join(dname, 'spk0', 'gt', fname)
        
        embeds = None                    # embedding
        if use_embed:                    #
            embeds = embed_dict[fpath]   # 
            embeds = embeds.unsqueeze(0).cuda()

        # ground truthのシンボリックリンクを貼る
        try:
            os.symlink(os.path.realpath(fpath), dst_path)
        except:
            pass

        with torch.no_grad():
            x_tst = text_norm.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([text_norm.size(0)]).cuda()
            # inference
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, embeds=embeds, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            scipy.io.wavfile.write(os.path.join(save_dir, fname), hps.data.sampling_rate, audio)

def _get_parser():
    parser = argparse.ArgumentParser(description='Inference evalation data.')
    parser.add_argument(
        'model_dir',
        type=str,
        help='Path of model directory',
    )
    return parser

# 与えられたモデルのconfig.jsonを返す
def _get_config(model_path):
    config_file = os.path.join(os.path.dirname(model_path), 'config.json')
    assert os.path.exists(config_file)
    return config_file

# テキストを所定のsequenceへ変換
def _get_normed_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# 予め保存していたembeddingベクトルを返す
def _get_embed_dict(embed_dir, test_files):
    test_filesname = os.path.basename(test_files)
    embed_file = os.path.join(embed_dir, test_filesname+'.pkl')
    assert os.path.exists(embed_file)
    with open(embed_file, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    main()