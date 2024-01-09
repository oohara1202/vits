# STUDIESモデルのみしか使えないスクリプト
# python inference_atr503.py logs/studies-teacher_serpp_ft
import os
import glob
import scipy
import torch

import utils
from models import SynthesizerTrn
from text.symbols import symbols

import inference_utils
from abelab_utils.g2p import G2P

ATR_DIR = 'atr503/full'
OUT_DIR = 'wav/atr503'
FEATURE_DIR = 'dump/averaged_vector_studies-teacher'  # for embedding vector models
emotion_list = ['Neutral', 'Happy', 'Sad']
emotion2id = {'Neutral':0, 'Happy':1, 'Sad':2}  # for a embedding table model

def main():
    args = inference_utils.get_parser().parse_args()
    model_dir = args.model_dir
    assert os.path.isdir(model_dir)

    # latestモデルのフルパス
    print('Model: ', end='')
    model_path = utils.latest_checkpoint_path(model_dir, 'G_*.pth')
    
    hps = utils.get_hparams_from_file(inference_utils.get_config(model_path))

    # wavファイルの保存directory作成
    d = os.path.basename(model_dir)
    assert len(d) != 0  # logs/jsut_base/ のようにスラッシュで終わっていると怒る
    dname = os.path.join(OUT_DIR, d)
    os.makedirs(dname, exist_ok=True)

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

    # 合成テキスト一覧を取得
    pp_symbols_dict = _get_pp_symbols_dict()

    sid = None                       # speaker id
    embeds = None                    # embedding
    embeds_ssl_lengths = None        #

    for emotion in emotion_list:
        print(f'Emotion: {emotion}')
        if not emotion == 'Sad':
            continue
        save_dir = os.path.join(dname, emotion)
        os.makedirs(save_dir, exist_ok=True)

        # Utterance-level feature embedding or SSL-model feature embedding
        # SSL-model featureもUtterance-levelになっている
        if use_embed or use_embed_ssl:
            feature_name = os.path.basename(model_dir).split('_')[1]
            feature_path = os.path.join(FEATURE_DIR, feature_name, f'{emotion}.pt')
            assert os.path.isfile(feature_path)
            embeds = torch.load(feature_path).unsqueeze(0).cuda()

        # Embedding table
        else:
            sid = emotion2id[emotion]
            sid = torch.LongTensor([int(sid)]).cuda()

        for fname, pp_symbols in pp_symbols_dict.items():
            text_norm = inference_utils.get_normed_text(pp_symbols, hps)

            with torch.no_grad():
                x_tst = text_norm.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([text_norm.size(0)]).cuda()
                # inference
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, embeds=embeds, embeds_ssl_lengths=embeds_ssl_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
                save_name = os.path.join(save_dir, fname+'.wav')
                scipy.io.wavfile.write(save_name, hps.data.sampling_rate, audio)
                # print(f'Emotion: {emotion}, Saved: {save_name}')

def _get_pp_symbols_dict():
    fcxlab_paths = sorted(glob.glob(os.path.join(ATR_DIR, '*.lab')))  # 503ファイル

    pp_symbols_dict = dict()
    for fcxlab_path in fcxlab_paths:
        bname = os.path.basename(fcxlab_path)
        pp_symbols = ' '.join(G2P.from_fcxlab(fcxlab_path))
        pp_symbols_dict[bname] = pp_symbols

    return pp_symbols_dict

if __name__ == "__main__":
    main()