# HTSから拝借したATR503文のファイル名をa01.lab形式変更
# http://hts.sp.nitech.ac.jp/?Download#zf6c5f7c

import os
import glob
import shutil

atr503_dirname = 'atr503'
types = ['full_original', 'mono_original']

for t in types:
    files = glob.glob(os.path.join(atr503_dirname, t, '*.lab'))
    type_name = t.split('_')[0]
    os.makedirs(os.path.join(atr503_dirname, type_name), exist_ok=True)
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        dstname = basename.split('_')[4] + '.lab'
        shutil.copyfile(f, os.path.join(atr503_dirname, type_name, dstname))
