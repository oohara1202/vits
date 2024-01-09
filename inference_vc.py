# python inference_vc.py logs/studies_vc
import os
import scipy
import torch

import utils
from models import SynthesizerTrn
from text.symbols import symbols
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch

import inference_utils

OUT_DIR = 'wav/vc'

def main():
    args = inference_utils.get_parser().parse_args()
    model_dir = args.model_dir

    # latestモデルのフルパス
    print('Model: ', end='')
    model_path = utils.latest_checkpoint_path(model_dir, 'G_*.pth')
    
    hps = utils.get_hparams_from_file(inference_utils.get_config(model_path))

    # testデータのfilelistのpathを取得
    test_files = hps.data.training_files.replace('_train_', '_test_')
    if 'studies-lite' in test_files:
        test_files = test_files.replace('studies-lite', 'studies')
        print(f'Used filelist: {test_files}')
    assert os.path.isfile(test_files)

    # wavファイルの保存directory作成
    d = os.path.basename(model_dir)
    assert len(d) != 0  # logs/jsut_base/ のようにスラッシュで終わっていると怒る
    dname = os.path.join(OUT_DIR, d)
    os.makedirs(dname, exist_ok=True)

    # 話者分directory作成
    for i in range(hps.data.n_speakers):
        for t in ['gt', 'gen']:  # ground truthと合成音声を比較する用
            os.makedirs(os.path.join(dname, f'spk{int(i)}', t), exist_ok=True)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers = hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)

    with open(test_files, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    
    sid_tgt = torch.LongTensor([0]).cuda()  # Teacher's id is 0

    for line in lines:
        line = line.rstrip()
        line_splitted = line.split('|')
        
        fpath = line_splitted[0]         # ファイルパス
        fname = os.path.basename(fpath)  # ファイル名
        sid = line_splitted[1]           # 話者id

        if sid == '0':  # Teacherのときはパス
            continue

        save_dir = os.path.join(dname, f'spk{sid}', 'gen')  
        dst_path = os.path.join(dname, f'spk{sid}', 'gt', fname)
        sid = torch.LongTensor([int(sid)]).cuda()

        # ground truthのシンボリックリンクを貼る
        try:
            os.symlink(os.path.realpath(fpath), dst_path)
        except:
            pass

        with torch.no_grad():
            spec = _get_spectrogram(fpath, hps).unsqueeze(0).cuda()
            spec_lengths = torch.LongTensor([spec.size(2)]).cuda()
            # inference
            audio = net_g.voice_conversion(spec, spec_lengths, sid_src=sid, sid_tgt=sid_tgt)[0][0,0].data.cpu().float().numpy()
            scipy.io.wavfile.write(os.path.join(save_dir, fname), hps.data.sampling_rate, audio)

# スペクトログラム算出（data_utils参考）
def _get_spectrogram(fpath, hps):
    audio, sampling_rate = load_wav_to_torch(fpath)
    if sampling_rate != hps.data.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, hps.data.sampling_rate))
    spec_filename = fpath.replace(".wav", ".spec.pt")
    if os.path.exists(spec_filename):
        spec = torch.load(spec_filename)
    else:
        audio_norm = audio / hps.data.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            center=False)
        spec = torch.squeeze(spec, 0)
    return spec

if __name__ == "__main__":
    main()
