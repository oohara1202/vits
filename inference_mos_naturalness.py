import os
import scipy
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

EXPERIMENT_DIR = '/work/abelab4/s_koha/s_koha_work/evaluation_experiment/mos_naturalness'
TEXT_DIR = os.path.join(EXPERIMENT_DIR, 'text_option')
OUT_DIR = os.path.join(EXPERIMENT_DIR, 'sppech_test')
CONVERTED_SPEECH_DIR = '/work/abelab4/s_koha/s_koha_work/evaluation_data/analysis_vc'

def main():
    emotion_list = ['Neutral', 'Happy', 'Sad']
    emotion2id = {'Neutral':0, 'Happy':1, 'Sad':2}  # for a embedding table model
    spkEn2spkJp = {'Teacher':'講師', 'FStudent':'女子生徒', 'MStudent':'男子生徒'}
    spkJp2spkEn = {v: k for k, v in spkEn2spkJp.items()}
    name_norm = {'FStudent':'F-student', 'MStudent':'M-student'}

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

    for emotion in emotion_list:
        save_dir = os.path.join(dname, emotion)
        os.makedirs(save_dir, exist_ok=True)

        textlist_path = os.path.join(TEXT_DIR, f'{emotion}.txt')
        assert os.path.exists(textlist_path)

        with open(textlist_path, mode='r', encoding='utf-8') as f:
            lines = [s.rstrip() for s in f.readlines()]
        
        for i in range(0, len(lines)-1, 4):
            txt_file = lines[i]  # 台本
            turn_num = lines[i+1].split(':')[1]
            student_name = spkJp2spkEn[lines[i+2].split('|')[0]]

            teacher_fxclab = txt_file.replace('/txt/', '/full/').replace('.txt', f'-Teacher-Turn-{turn_num}.lab')
            assert os.path.exists(teacher_fxclab)
            filename = os.path.splitext(os.path.basename(teacher_fxclab))[0]

            sid = None                       # speaker id
            embeds = None                    # embedding
            embeds_ssl_lengths = None        #

            # x-vector or SSL-model feature embedding
            if use_embed or use_embed_ssl:
                n = os.path.splitext(os.path.basename(teacher_fxclab))[0]
                basename = n.replace('Teacher', student_name) + '.wav'
                reference_path = os.path.join(CONVERTED_SPEECH_DIR, f'{name_norm[student_name]}_converted', basename)
                # reference_path = os.path.join(CONVERTED_SPEECH_DIR, 'Teacher', n+'.wav')
                assert reference_path

                embeds = extractor(reference_path)
                embeds = embeds.cuda().unsqueeze(0)
                if use_embed_ssl:
                    embeds_ssl_lengths = torch.LongTensor(1).cuda()
                    embeds_ssl_lengths[0] = embeds.size(1)

            # Embedding table
            else:
                sid = emotion2id[emotion]
                sid = torch.LongTensor([int(sid)]).cuda()

            # テキスト処理
            pp_symbols = ' '.join(G2P.from_fcxlab(teacher_fxclab))
            with torch.no_grad():
                text_norm = inference_utils.get_normed_text(pp_symbols, hps)
                x_tst = text_norm.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([text_norm.size(0)]).cuda()
                # inference
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, embeds=embeds, embeds_ssl_lengths=embeds_ssl_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
                save_path = os.path.join(save_dir, f'{filename}.wav')
            scipy.io.wavfile.write(save_path, hps.data.sampling_rate, audio)

if __name__ == "__main__":
    main()
