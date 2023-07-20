import os
import glob
from typing import Iterable, List, Optional, Union

from g2p import G2P

def main():
    ############################################
    # ここを変える
    calls_dirname = 'CALLS'
    # validationとtestのディレクトリ名
    VAL_DIRS = []
    TEST_DIRS = []
    dst_dirname = 'filelists/calls'        # 出力ディレクトリ
    dst_filename = 'calls_audio_sid_text'  # 出力ファイル名（一部）
    emotion2id = {'平静':0, '喜び':1, '悲しみ':2}  # 感情-->ID
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    calls_dir = os.path.join(basedir, calls_dirname)
    assert os.path.isdir(calls_dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # 出力するファイルリスト
    filelist = dict()
    filelist['val'] = list()
    filelist['test'] = list()
    filelist['train'] = list()

    for type_name in ['P-dialogue', 'S-dialogue']:
        type_dir = os.path.join(calls_dir, type_name)
        # P01, S01などのディレクトリ名を取得
        dir_list = [f for f in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, f))]

        for dname in dir_list:
            if dname in VAL_DIRS:
                tr_name = 'val'
            elif dname in TEST_DIRS:
                tr_name = 'test'
            else:
                tr_name = 'train'
            
            d = os.path.join(type_dir, dname)

            files = list()  # 一旦保存
            txt_files = sorted(glob.glob(os.path.join(d, '**/txt/*.txt'), recursive=True))
            wav_files = sorted(glob.glob(os.path.join(d, f'**/wav/*.wav'), recursive=True))
            i = 0
            for txt_file in txt_files:
                with open(txt_file, mode='r', encoding='utf-8') as f:
                    lines = f.readlines()
                lines = [s for s in lines if s.split('|')[0]=='オペレータ']
                for line in lines:
                    emotion = line.split('|')[1]  # 感情
                    
                    # 怒り（Angry）は除外
                    if emotion == '怒り':
                        continue
                    
                    text = line.split('|',)[2]  # 平文
                    text = text.replace('\u3000', '')  # 全角スペースを削除
                    filepath = wav_files[i]

                    newline = f'{filepath}|{emotion2id[emotion]}|{text}\n'
                    files.append(newline)

                    i+=1

            filelist[tr_name].extend(files)

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])

        # G2P後に保存
        savename = savename + '.cleaned'
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            filelist_cleaned = list()
            for line in filelist[t]:
                line = line.rstrip()
                filepath, spkid, grapheme = line.split('|')
                # フルコンテキストラベルのpathに置換
                fcxlab_path = filepath.replace('/wav/', '/full/').replace('.wav', '.lab')
                assert os.path.isfile(fcxlab_path)

                symbols = ' '.join(G2P.from_fcxlab(fcxlab_path))
                # 再構成
                newline = f'{filepath}|{spkid}|{symbols}\n'
                filelist_cleaned.append(newline)

            f.writelines(filelist_cleaned)

if __name__ == '__main__':
    main()
