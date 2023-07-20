# Commands

## JSTU training

```bash
# preprocess dataset
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/jsut_ver1.1-p1_22k jsut_ver1.1

# (optional) extract x-vector
python prep_dataset/11_save_xvector.py filelists/jsut

# train
## original
python train.py -c configs/jsut_base.json -m jsut_base
## with x-vector
python train_xvector.py -c configs/jsut_xvector.json -m jsut_xvector
```

## JVS training

```bash
# preprocess dataset
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/jvs_ver1_22k jvs_ver1
python prep_dataset/01_prep_jsut.py

# (optional) extract x-vector
python prep_dataset/11_save_xvector.py filelists/jvs

# train
## original
python train_ms.py -c configs/jvs_base.json -m jvs_base
## with x-vector
python train_xvector.py -c configs/jvs_xvector.json -m jvs_xvector
```

## STUDIES (Teacher model) training

```bash
# preprocess dataset
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/STUDIES_22k STUDIES
python prep_dataset/03_prep_studies-teacher.py

# (optional) extract x-vector
python prep_dataset/11_save_xvector.py filelists/studies-teacher

# train
## with x-vector
python train_xvector.py -c configs/studies-teacher_xvector.json -m studies-teacher_xvector

# finetune
TODO
```

## STUDIES + CALLS training

```bash
# preprocess STUDIES
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/STUDIES_22k STUDIES
python prep_dataset/03_prep_studies-teacher.py

# preprocess CALLS
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/STUDIES-2_22k CALLS
python prep_dataset/04_prep_calls.py

# integrate two datasets
python prep_dataset/21_integrate_studies_calls.py

# (optional) extract x-vector
python prep_dataset/11_save_xvector.py filelists/studies-calls

# train
## with x-vector
python train_xvector.py -c configs/studies-calls_xvector.json -m studies-calls_xvector
```

## LJSpeech training

``` bash
# preprocess dataset
mkdir DUMMY1
ln -s /abelab/DB4/LJSpeech-1.1/wavs/*.wav DUMMY1/

# train
python train.py -c configs/ljs_base.json -m ljs_base
```

## Exporting Conda's environment

```bash
conda env export --no-build -n vits > requirements.yml
```
