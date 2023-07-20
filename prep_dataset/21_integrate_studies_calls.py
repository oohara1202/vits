# STUDIESとCALLSを統合したfilelistsを作成
# STUDIESはTeacherのみの点に注意
import os
import glob
from typing import Iterable, List, Optional, Union

def main():
    ############################################
    # ここを変える
    studies_list = 'filelists/studies-teacher'  # 統合元
    calls_list = 'filelists/calls'              # 統合元
    dst_dirname = 'filelists/studies-calls'        # 出力ディレクトリ
    dst_name = 'studies-calls'  # 出力ファイル名（一部）
    ############################################

    # 'prep_dataset'内では実行しない
    assert os.path.basename(os.getcwd()) != 'prep_dataset'

    os.makedirs(dst_dirname, exist_ok=True)

    txt_studies = sorted(glob.glob(os.path.join(studies_list, '*')))
    txt_calls = sorted(glob.glob(os.path.join(calls_list, '*')))

    for st, ct in zip(txt_studies, txt_calls):
        rest = st.split('_', 1)[1]  # コーパス名を分離
        savename = os.path.join(dst_dirname, f'{dst_name}_{rest}')

        with open(st, mode='r', encoding='utf-8') as sf:      # STUDIES
            with open(ct, mode='r', encoding='utf-8') as cf:  # CALLS
                with open(os.path.join(savename), mode='w', encoding='utf-8') as f:
                    f.write(sf.read())
                    f.write(cf.read())

if __name__ == '__main__':
    main()
