# python prep_dataset/31_averaged_vector.py xvector
import os
import glob
import argparse
import pickle

import torch
from torch import nn
from torch.nn import functional as F

studies_dir = 'dataset/STUDIES'
filelists_dir = 'filelists/studies-teacher'
dump_root = 'dump'
dump_dir = 'averaged_vector_studies-teacher'
weights_path = 'dump/averaged_vector_studies-teacher/weighted_sum_weights.pt'  # for ssl

enSpk2jpSpk = {'Teacher':'講師', 'FStudent':'女子生徒', 'MStudent':'男子生徒'}
emotion_list = ['Neutral', 'Happy', 'Sad']
enEmo2jpEmo = {'平静': 'Neutral', '喜び': 'Happy', '悲しみ': 'Sad', '怒り': 'Angry'}

def main():
    assert os.path.isdir(studies_dir)

    args = _get_parser().parse_args()
    feature_name = args.feature

    feature_dir = os.path.join(dump_root, feature_name, 'studies-teacher')
    assert os.path.isdir(feature_dir)

    # ファイル名をkeyに感情を記録
    basename2emotion = _get_basename2emotion(studies_dir)

    feature_emotion_dict = dict()
    for e in emotion_list:
        feature_emotion_dict[e] = list()

    filelists_list = glob.glob(os.path.join(filelists_dir, '*.cleaned'))

    # SSL-model feature
    if feature_name == 'sslfeature':
        ulf = UtteranceLevelFeaturizer()
        for filelists in filelists_list:
            # 音声ごとに保存してあるSSLモデル特徴量
            feature_files = glob.glob(os.path.join(feature_dir, os.path.basename(filelists), '*.pkl'))
            assert len(feature_files) > 1

            for feature_file in feature_files:
                # 読み込み
                with open(feature_file, 'rb') as f:
                    # [layer(13), frame(any), feature(768)]
                    embeds_ssl = pickle.load(f).cpu()
                # [batch(1), layer(13), frame(any), feature(768)]
                embeds_ssl = embeds_ssl.unsqueeze(0)
                with torch.no_grad():
                    # Utterance-levelへ
                    feature = ulf(embeds_ssl)

                # basenameと.pkl外し
                basename = os.path.splitext(os.path.basename(feature_file))[0]
                emotion = basename2emotion[basename]  # 感情

                feature_emotion_dict[emotion].append(F.normalize(feature, dim=0))
    
    # Uttrance-level feature
    else:
        for filelists in filelists_list:
            # dictで保存してある特徴量
            feature_dict_path = os.path.join(feature_dir, os.path.basename(filelists)+'.pkl')
            assert os.path.isfile(feature_dict_path)

            with open(feature_dict_path, 'rb') as f:
                feature_dict = pickle.load(f)

            for filepath, feature in feature_dict.items():
                '''
                filepath: ファイルパス
                feature : 特徴量（torch.Tensor）
                '''
                if 'dataset/STUDIES' in filepath and 'Teacher' in filepath:
                    basename = os.path.basename(filepath)
                    emotion = basename2emotion[basename]  # 感情

                    feature_emotion_dict[emotion].append(F.normalize(feature, dim=0))
    
    save_dir = os.path.join(dump_root, dump_dir, feature_name)
    os.makedirs(save_dir, exist_ok=True)
    for e in emotion_list:
        print(f'{e} file: {len(feature_emotion_dict[e])} files')
        feature_stack = torch.stack(feature_emotion_dict[e], dim=0)
        # calculate averaged feature vector
        avg_feature = torch.mean(feature_stack, dim=0)
        # save
        torch.save(avg_feature, os.path.join(save_dir, f'{e}.pt'))
        # print(avg_feature)
    print(f'Feature shape: {avg_feature.size(0)}')

def _get_basename2emotion(studies_dir):
    basename2emotion = dict()

    for type_name in ['ITA', 'Long_dialogue', 'Short_dialogue']:
        type_dir = os.path.join(studies_dir, type_name)
        
        # Emotion100-Angry, LD01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]
        for dname in dir_list:
            d = os.path.join(type_dir, dname)

            for spk in ['Teacher', 'MStudent', 'FStudent']:
                txt_files = sorted(glob.glob(os.path.join(d, '**/txt/*.txt'), recursive=True))
                wav_files = sorted(glob.glob(os.path.join(d, f'**/wav/*{spk}*.wav'), recursive=True))
                i = 0
                for txt_file in txt_files:
                    with open(txt_file, mode='r', encoding='utf-8') as f:
                        lines = f.readlines()
                    lines = [s for s in lines if s.split('|')[0]==enSpk2jpSpk[spk]]
                    for line in lines:
                        emotion = line.split('|')[1]  # 感情

                        filepath = wav_files[i]
                        basename = os.path.basename(filepath)

                        basename2emotion[basename] = enEmo2jpEmo[emotion]

                        i+=1

    return basename2emotion

def _get_parser():
    parser = argparse.ArgumentParser(description='Feature name')
    parser.add_argument(
        'feature',
        type=str,
        help='Feature name',
    )
    return parser

# models.UtteranceLevelFeaturizerに基づく
class UtteranceLevelFeaturizer(nn.Module):
  def __init__(self, layer_num=13):
    super().__init__()
    self.layer_num = layer_num
    self.weights = torch.load(weights_path)  # 
    print(f'Use weighted-sum weights:{self.weights}')

  def _weighted_sum(self, embeds_ssl):  # [batch(1), layer(13), frame(any), feature(768)]
    embeds_ssl = embeds_ssl.transpose(0, 1)  # [layer(13), batch(1), frame(any), feature(768)]
    _, *origin_shape = embeds_ssl.shape
    embeds_ssl = embeds_ssl.contiguous().view(self.layer_num, -1)  # [layer(13), any(any)]
    norm_weights = F.softmax(self.weights, dim=-1)
    weighted_embeds_ssl = (norm_weights.unsqueeze(-1) * embeds_ssl).sum(dim=0)  # [any(any)]
    weighted_embeds_ssl = weighted_embeds_ssl.view(*origin_shape)  # [batch(1), frame(any), feature(768)]
   
    return weighted_embeds_ssl

  def forward(self, embeds_ssl):
    embeds_ssl_BxTxH = self._weighted_sum(embeds_ssl)
    
    # averaged pooling
    embeds_ssl_TxH = embeds_ssl_BxTxH.squeeze()
    embeds_ssl = torch.mean(embeds_ssl_TxH, dim=0)

    return embeds_ssl

if __name__ == '__main__':
    main()
