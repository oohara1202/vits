import os
from typing import Iterable, List, Optional, Union

from utils import text_clean_multi

def main():
    ############################################
    # ここを変える
    jvs_dirname = 'jvs_ver1'  
    VAL_NUM = 2   # validationとtestのデータ数
    TEST_NUM = 5  #
    dst_dirname = 'filelists/jvs'        # 出力ディレクトリ
    dst_filename = 'jvs_audio_sid_text'  # 出力ファイル名（一部）
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    jvs_dir = os.path.join(basedir, jvs_dirname)
    assert os.path.isdir(jvs_dir)

    os.makedirs(dst_dirname, exist_ok=True)

    # jvs001, jvs002などのディレクトリ名を取得
    dir_list = [f for f in os.listdir(jvs_dir) if os.path.isdir(os.path.join(jvs_dir, f))]
    dir_list.sort()

    # 出力するファイルリスト
    filelist = dict()
    filelist['val'] = list()
    filelist['test'] = list()
    filelist['train'] = list()

    for spkid, dname in enumerate(dir_list):  # 話者ごとに回す
        d = os.path.join(jvs_dir, dname)

        files = list()  # 話者ごとに一旦保存        
        for utter in ['nonpara30', 'parallel100']:  # nonpara30の上からn音声をvalとtestにするため
            # 台本ファイル
            # 台本ファイルの発話数と実際のwavファイル数が異なる点に注意
            with open(os.path.join(d, utter, 'transcripts_utf8.txt'), mode='r', encoding='utf-8') as f:
                for line in f:
                    # 拡張子なしファイル名をpathへ変換
                    name, text = line.split(':')
                    filepath = os.path.join(d, utter, 'wav24kHz16bit', f'{name}.wav')
                    if os.path.isfile(filepath):
                        assert os.path.isfile(filepath)
                        newline = f'{filepath}|{spkid}|{text}'
                        files.append(newline)
        
        filelist['val'].extend(files[:VAL_NUM])
        filelist['test'].extend(files[VAL_NUM:VAL_NUM+TEST_NUM])
        filelist['train'].extend(files[VAL_NUM+TEST_NUM:])

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])

        # G2P後に保存
        savename = savename + '.cleaned'
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(text_clean_multi(filelist[t]))

if __name__ == '__main__':
    main()
