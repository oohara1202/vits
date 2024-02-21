import os
import scipy
import datetime
import torch

# 雑な対処
import logging
logging.getLogger('speechbrain').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

import utils
from models import SynthesizerTrn
from text.symbols import symbols

from abelab_utils.g2p import G2P
import inference_utils

OUT_DIR = 'wav/free'

def main():
    args = inference_utils.get_parser().parse_args()
    model_dir = args.model_dir

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

    # 特徴量抽出器
    if use_embed:
        from abelab_utils.extract_xvector import ExtractXvector
        extractor = ExtractXvector()
    elif use_embed_ssl:
        from abelab_utils.extract_sslfeature import ExtractSSLModelFeature
        extractor = ExtractSSLModelFeature()

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

    sid = None                       # speaker id
    embeds = None                    # embedding
    embeds_ssl_lengths = None        #

    # 合成を繰り返す
    while True:
        save_name = list()

        # フルコンテキストラベルファイルのパスが与えられたとき
        if args.fcxlab:
            # テキスト処理
            pp_symbols = ' '.join(G2P.from_fcxlab(args.fcxlab))
            print(f'Phoneme: {pp_symbols}')
            save_name.append(os.path.splitext(os.path.basename(args.fcxlab))[0])

        else:
            # テキスト入力
            print('Input text. (Exit: \"q\")')
            text = input()
            if text == 'q':   
                break
            pp_symbols = ' '.join(G2P.from_grapheme(text))
            print(f'Phoneme: {pp_symbols}')
            save_name.append(text)

        # multi-speaker settingの場合は話者を選択
        if hps.data.n_speakers > 0 and not use_embed and not use_embed_ssl:
            while True:
                print('Input speaker ID. (Exit: \"q\")')
                sid = input()
                if sid == 'q':
                    exit()
                # 数値で受け付けているか
                if sid.isdecimal():
                    sid = int(sid)
                    # 存在するIDか
                    if sid < int(hps.data.n_speakers):
                        save_name.append(f'spk{sid:02}')
                        sid = torch.LongTensor([sid]).cuda()
                        break
                    else:
                        print(f'Out of range. (Note: 0-{hps.data.n_speakers})')
                else:
                    print('Invalid format.')

        elif args.embed:
            embeds = torch.load(args.embed).unsqueeze(0).cuda()
            print(f'Embedding dim: {embeds.shape}')
            save_name.append(os.path.splitext(os.path.basename(args.embed))[0])

        else:
            print('Input path of reference. (Exit: \"q\")')
            reference_path = input()
            if reference_path == 'q':
                break
            embeds = extractor(reference_path)
            embeds = embeds.cuda().unsqueeze(0)
            if use_embed_ssl:
                embeds_ssl_lengths = torch.LongTensor(1).cuda()
                embeds_ssl_lengths[0] = embeds.size(1)
            save_name.append(os.path.splitext(os.path.basename(reference_path))[0])

        date = datetime.datetime.now().strftime('%m%d-%H%M%S')
        save_name.insert(0, date)

        with torch.no_grad():
            text_norm = inference_utils.get_normed_text(pp_symbols, hps)
            x_tst = text_norm.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([text_norm.size(0)]).cuda()
            # inference
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, embeds=embeds, embeds_ssl_lengths=embeds_ssl_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            save_path = os.path.join(dname, '_'.join(save_name)+'.wav')
            scipy.io.wavfile.write(save_path, hps.data.sampling_rate, audio)
            print(f'Saved: {save_path}')

        # 事前に入力が与えられているときは繰り返さない
        if args.fcxlab or args.embed:
            break

if __name__ == "__main__":
    main()
