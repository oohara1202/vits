# 橋本くん用合成スクリプト
# 書き換えないで！
import os
import glob
import scipy
import pickle
import torch

import utils
from models import SynthesizerTrn
from text.symbols import symbols

import inference_utils
from abelab_utils.g2p import G2P

emotion_list = ['Neutral', 'Happy', 'Sad']
emotion2id = {'Neutral':0, 'Happy':1, 'Sad':2}  # for a embedding table model

MODEL_DIR = 'logs/studies-teacher_xvector_ft'
ATR_DIR = 'atr503/full'
OUT_DIR = '/work/abelab5/r_hashi/r_hashi_work/dump/wav'
XVECTOR_DIR = '/work/abelab5/r_hashi/r_hashi_work/dump/avg_xvector/withoutITA'

def main():
    assert os.path.isdir(MODEL_DIR)

    # latestモデルのフルパス
    print('Model: ', end='')
    model_path = utils.latest_checkpoint_path(MODEL_DIR, 'G_*.pth')
    
    hps = utils.get_hparams_from_file(inference_utils.get_config(model_path))

    #################################################
    # モデル立ち上げ
    use_embed = hps.data.use_embed
    use_embed_ssl = False
    embed_dim = hps.data.embed_dim

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

    # 合成テキスト一覧を取得
    pp_symbols_dict = _get_pp_symbols_dict()

    sid = None                       # speaker id
    embeds = None                    # embedding
    embeds_ssl_lengths = None        #
    #################################################

    # クラスターごとの平均x-vector共
    xvector_pkls = sorted(glob.glob(os.path.join(XVECTOR_DIR, '*.pkl')))

    for xvector_pkl in xvector_pkls:
        # n_clustersに合わせてdirectory作成
        d = os.path.splitext(os.path.basename(xvector_pkl))[0]
        dname = os.path.join(OUT_DIR, d)
        os.makedirs(dname, exist_ok=True)
        
        # 平均x-vectorのdict読み込み
        with open(xvector_pkl, 'rb') as f:
            avg_xvector_dict = pickle.load(f)
        
        # 各クラスターの平均x-vectorで合成
        for cluster_idx, avg_xvector in avg_xvector_dict.items():
            # wavファイルの保存directory作成
            save_dir = os.path.join(dname, cluster_idx)
            os.makedirs(save_dir, exist_ok=True)

            embeds = avg_xvector.unsqueeze(0).cuda()

            for fname, pp_symbols in pp_symbols_dict.items():
                text_norm = inference_utils.get_normed_text(pp_symbols, hps)

                with torch.no_grad():
                    x_tst = text_norm.cuda().unsqueeze(0)
                    x_tst_lengths = torch.LongTensor([text_norm.size(0)]).cuda()
                    # inference
                    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, embeds=embeds, embeds_ssl_lengths=embeds_ssl_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
                    save_name = os.path.join(save_dir, fname+'.wav')
                    scipy.io.wavfile.write(save_name, hps.data.sampling_rate, audio)
        print(f'Synthesized: {d}')

def _get_pp_symbols_dict():
    fcxlab_paths = sorted(glob.glob(os.path.join(ATR_DIR, '[aj]*.lab')))  # 503ファイル

    pp_symbols_dict = dict()
    for fcxlab_path in fcxlab_paths:
        bname = os.path.basename(fcxlab_path)
        pp_symbols = ' '.join(G2P.from_fcxlab(fcxlab_path))
        pp_symbols_dict[bname] = pp_symbols

    return pp_symbols_dict

if __name__ == "__main__":
    main()
