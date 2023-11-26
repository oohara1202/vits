# filelistsディレクトリに従ってSSLモデル特徴量を抽出・保存
# frame-level特徴量であることに注意！
# 次元数デカすぎてファイルごとに保存
import os
import argparse
import glob
import pickle

from extract_sslfeature import ExtractSSLFeature

def main():
    ############################################
    # ここを変える
    dump_dirname = 'dump/sslfeature'
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

    extractor = ExtractSSLFeature()

    for cleaned_file in cleaned_files:
        save_dirname = os.path.join(dump_dirname, os.path.basename(cleaned_file))
        os.makedirs(save_dirname, exist_ok=True)  
        print(f'Extracting ssl model feature: {cleaned_file}')
        with open(cleaned_file, encoding='utf-8') as f:
            for line in f:
                audiopath = line.strip().split('|')[0]

                sslfeature = extractor(audiopath)  # 長さ13のtuple，中身はtorch.Size([1, frame, 768])

                # SSLモデル特徴量を「ファイルごとに」pickleで保存
                save_sslfeature(
                    os.path.join(save_dirname, os.path.basename(audiopath)+'.pkl'),
                    sslfeature
                )

            print(f'Saved ssl model feature: {cleaned_file}')

# pickleで保存     
def save_sslfeature(save_path, sslfeature):
    with open(save_path, 'wb') as f:
        pickle.dump(sslfeature, f)

def get_parser():
    parser = argparse.ArgumentParser(description='Extract and save ssl model features.')
    parser.add_argument(
        'dirpath',
        type=str,
        help='Path of filelists directory',
    )
    return parser

if __name__ == '__main__':
    main()
