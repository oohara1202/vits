#!/usr/bin/env bash
# 大本のディレクトリ構造を保持しつつ，ファイル群のシンボリックリンクを貼る

# check current directory
if [ $(basename $PWD) == "prep_dataset" ]; then
    echo "Move up to higher level directory e.g. vits"
fi
# check arguments
if [ $# != 2 ]; then
    echo "Usage: $0 <db> <dirname>"
    echo "e.g.: $0 /work/abelab4/s_koha/s_koha_work/dataset/jsut_ver1.1-p1_22k jsut_ver1.1"
    exit 1
fi

SRC_DIR=$1
DST_DIR=./dataset/$2

(cd ${SRC_DIR}; find * -type d) | xargs -I _DIR_ mkdir -p ${DST_DIR}/_DIR_

for f in $(cd ${SRC_DIR}; find * -type f ); do
    f=$(dirname $f)/$(basename $f)
    ln -s ${SRC_DIR}/$f ${DST_DIR}/$f
done
