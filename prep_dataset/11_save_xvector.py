# filelistsディレクトリに従ってx-vectorを抽出・保存
import os
import argparse
import glob
import pickle

from extract_xvector import ExtractXvector

def main():
    ############################################
    # ここを変える
    dump_dirname = 'dump/xvector'
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    args = get_parser().parse_args()
    assert os.path.isdir(args.dirpath)

    # ".cleaned"のみglob
    cleaned_files = glob.glob(os.path.join(args.dirpath, '*.cleaned'))
    assert len(cleaned_files) > 0

    # 保存ディレクトリ作成
    dump_dirname = os.path.join(dump_dirname, os.path.basename(args.dirpath))
    os.makedirs(dump_dirname, exist_ok=True)

    extractor = ExtractXvector()

    for cleand_files in cleaned_files:
        xvector_dict = dict()  # audiopathをkeyにx-vectorを保存

        print(f'Extracting x-vector: {cleand_files}')
        with open(cleand_files, encoding='utf-8') as f:
            for line in f:
                audiopath = line.strip().split('|')[0]
                
                xvector = extractor(audiopath)  # 抽出
                xvector_dict[audiopath] = xvector

        # pickleで保存
        save_path = os.path.join(dump_dirname, os.path.basename(cleand_files)+'.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(xvector_dict, f)
        print(f'Saved x-vector: {save_path}')

def get_parser():
    parser = argparse.ArgumentParser(description='Extract and save x-vectors.')
    parser.add_argument(
        'dirpath',
        type=str,
        help='Path of filelists directory',
    )
    return parser

if __name__ == '__main__':
    main()
