import os
from typing import Iterable, List, Optional, Union

from utils import text_clean_single

def main():
    ############################################
    # ここを変える
    jsut_dirname = 'jsut_ver1.1'  
    VAL_NUM = 250   # validationとtestのデータ数
    TEST_NUM = 250  #
    dst_dirname = 'filelists'          # 出力ディレクトリ
    dst_filename = 'jsut_audio_text'   # 出力ファイル名（一部）
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    basedir = 'dataset'
    jsut_dir = os.path.join(basedir, jsut_dirname)
    assert os.path.isdir(jsut_dir)

    # basic5000, countersuffix26などのディレクトリ名を取得
    dir_list = [f for f in os.listdir(jsut_dir) if os.path.isdir(os.path.join(jsut_dir, f))]
    dir_list.sort()

    # 出力するファイルリスト
    filelist = dict()

    files = list()
    for dname in dir_list:
        d = os.path.join(jsut_dir, dname)

        # 台本ファイル
        with open(os.path.join(d, 'transcript_utf8.txt'), mode='r', encoding='utf-8') as f:
            for line in f:
                # 拡張子なしファイル名をpathへ変換
                name, text = line.split(':')
                filepath = os.path.join(d, 'wav', f'{name}.wav')
                
                newline = f'{filepath}|{text}'
                files.append(newline)

    filelist['val'] = files[:VAL_NUM]
    filelist['test'] = files[VAL_NUM:VAL_NUM+TEST_NUM]
    filelist['train'] = files[VAL_NUM+TEST_NUM:]

    for t in ['val', 'test', 'train']:
        savename = os.path.join(dst_dirname, f'{dst_filename}_{t}_filelist.txt')
        
        # G2P前で保存
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(filelist[t])

        # G2P後に保存
        savename = savename + '.cleaned'
        with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
            f.writelines(text_clean_single(filelist[t]))

if __name__ == '__main__':
    main()
