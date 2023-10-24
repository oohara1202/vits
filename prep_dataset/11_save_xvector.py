# filelistsディレクトリに従ってx-vectorを抽出・保存
# ついでに話者ごとに平均x-vectorを出す <-- これのせいでプログラムに品がない
import os
import argparse
import glob
import pickle
import torch

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
    # 平均x-vector用フォルダまで一気に作る
    spk_dirname = os.path.join(dump_dirname, 'speaker')
    os.makedirs(spk_dirname, exist_ok=True)  

    extractor = ExtractXvector()

    if '_audio_sid_text_' not in cleaned_files[0]:  # single-spk dataset
        print(f'Corpus: {args.dirpath} is \"SINGLE\" speaker dataset.')
        for cleaned_file in cleaned_files:
            xvector_dict = dict()  # audiopathをkeyにx-vectorを保存
            xvector_list = list()  # 平均x-vector算出用

            print(f'Extracting x-vector: {cleaned_file}')
            with open(cleaned_file, encoding='utf-8') as f:
                for line in f:
                    audiopath = line.strip().split('|')[0]

                    xvector = extractor(audiopath)  # 抽出
                    
                    xvector_dict[audiopath] = xvector
                    xvector_list.append(xvector)

            # x-vectorをpickleで保存
            save_xvector(
                os.path.join(dump_dirname, os.path.basename(cleaned_file)+'.pkl'),
                xvector_dict
            )

            # 平均x-vectorをtorch.saveで保存
            save_spk_xvector(
                os.path.join(spk_dirname, 'spk0_'+os.path.basename(cleaned_file)+'.pt'),
                xvector_list
            )

            print(f'Saved x-vector: {cleaned_file}')

    else:  # multi-spk dataset
        print(f'Corpus: {args.dirpath} is \"MULTI\" speaker dataset.')
        for cleaned_file in cleaned_files:
            xvector_dict = dict()        # audiopathをkeyにx-vectorを保存
            xvector_list_dict = dict()  # 平均x-vector算出用（複数話者のためにdict）

            print(f'Extracting x-vector: {cleaned_file}')
            with open(cleaned_file, encoding='utf-8') as f:
                for line in f:
                    audiopath = line.strip().split('|')[0]
                    sid = line.strip().split('|')[1]

                    xvector = extractor(audiopath)  # 抽出

                    xvector_dict[audiopath] = xvector
                    # 辞書を用いてsidごとにx-vectorを貯める
                    # sidをkeyにしてdictを作り，sidごとにlistにappendする処理
                    # 初回のlist宣言のためにtry-catch
                    try:
                        xvector_list_dict[sid].append(xvector)
                    except KeyError:
                        xvector_list_dict[sid] = list()
                        xvector_list_dict[sid].append(xvector)
            
            # x-vectorをpickleで保存
            save_xvector(
                os.path.join(dump_dirname, os.path.basename(cleaned_file)+'.pkl'),
                xvector_dict
            )

            # 話者ごとに平均x-vectorを算出・保存
            for sid in xvector_list_dict:
                # 平均x-vectorをtorch.saveで保存
                filename = 'spk{}_{}.pt'.format(sid, os.path.basename(cleaned_file))
                save_spk_xvector(
                    os.path.join(spk_dirname, filename),
                    xvector_list_dict[sid]
                )

            print(f'Saved x-vector: {cleaned_file}')

# pickleで保存     
def save_xvector(save_path, xvector_dict):
    with open(save_path, 'wb') as f:
        pickle.dump(xvector_dict, f)

# torch.saveで保存
def save_spk_xvector(save_path, xvector_list):
    num_xvectror = len(xvector_list)
    print(f'{num_xvectror} x-vectors are used for calculating the averaged x-vector:')
    print(f'  {save_path}')
    # speaker normalization
    averaged_xvector = torch.mean(torch.stack(xvector_list, dim=0), dim=0)
    torch.save(averaged_xvector, save_path)

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
