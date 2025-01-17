# Commands

## JSUT training

```bash
# preprocess dataset
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/jsut_ver1.1-p1_22k jsut_ver1.1
python prep_dataset/01_prep_jsut.py

# (optional) extract x-vector
python prep_dataset/11_save_xvector.py filelists/jsut

# train
## original
nohup python train.py -c configs/jsut_base.json -m jsut_base > logs/jsut_base.out
## with x-vector
nohup python train_embed.py -c configs/jsut_xvector.json -m jsut_xvector > logs/jsut_xvector.out
```

## JVS training

```bash
# preprocess dataset
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/jvs_ver1_22k jvs_ver1
python prep_dataset/02_prep_jvs.py

# (optional) extract x-vector
python prep_dataset/11_save_xvector.py filelists/jvs
# (optional) extract SSL model feature
python prep_dataset/12_save_sslfeature.py filelists/jvs

# train
## original
nohup python train_ms.py -c configs/jvs_base.json -m jvs_base > logs/jvs_base.out
## with x-vector
nohup python train_embed.py -c configs/jvs_xvector.json -m jvs_xvector > logs/jvs_xvector.out
## with SSL model feature
nohup python train_embed_ssl.py -c configs/jvs_sslfeature.json -m jvs_sslfeature > logs/jvs_sslfeature.out
```

## STUDIES + CALLS training

```bash
# preprocess STUDIES-theacher
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
# (optional) extract SSL model feature
python prep_dataset/12_save_sslfeature.py filelists/studies-calls
# (optional) extract SER posterior probability and Emotion representation
## Please check "../s3prl" programs

# train
## original
nohup python train_ms.py -c configs/studies-calls_base.json -m studies-calls_base > logs/studies-calls_base.out
## with x-vector
nohup python train_embed.py -c configs/studies-calls_xvector.json -m studies-calls_xvector > logs/studies-calls_xvector.out
## with SSL model feature
nohup python train_embed_ssl.py -c configs/studies-calls_sslfeature.json -m studies-calls_sslfeature > logs/studies-calls_sslfeature.out
## with SER posterior probability
nohup python train_embed.py -c configs/studies-calls_serpp.json -m studies-calls_serpp > logs/studies-calls_serpp.out
## with Emotion representation
nohup python train_embed.py -c configs/studies-calls_emorepresentation.json -m studies-calls_emorepresentation > logs/studies-calls_emorepresentation.out
## with SER posterior probability (JTES SER model)
nohup python train_embed.py -c configs/studies-calls_jserpp.json -m studies-calls_jserpp > logs/studies-calls_jserpp.out
## with Emotion representation (JTES SER model)
nohup python train_embed.py -c configs/studies-calls_jemorepresentation.json -m studies-calls_jemorepresentation > logs/studies-calls_jemorepresentation.out
```

## STUDIES (female-teacher model) training

```bash
# preprocess dataset
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/STUDIES_22k STUDIES
python prep_dataset/03_prep_studies-teacher.py

# (optional) extract x-vector
python prep_dataset/11_save_xvector.py filelists/studies-teacher
# (optional) extract SSL model feature
python prep_dataset/12_save_sslfeature.py filelists/studies-teacher

# train
## original
nohup python train_ms.py -c configs/studies-teacher_base.json -m studies-teacher_base > logs/studies-teacher_base.out
## with x-vector
nohup python train_embed.py -c configs/studies-teacher_xvector.json -m studies-teacher_xvector > logs/studies-teacher_xvector.out
## with SSL model feature
nohup python train_embed_ssl.py -c configs/studies-teacher_sslfeature.json -m studies-teacher_sslfeature > logs/studies-teacher_sslfeature.out
## with SER posterior probability
nohup python train_embed.py -c configs/studies-teacher_serpp.json -m studies-teacher_serpp > logs/studies-teacher_serpp.out
## with Emotion representation
nohup python train_embed.py -c configs/studies-teacher_emorepresentation.json -m studies-teacher_emorepresentation > logs/studies-teacher_emorepresentation.out

# fine-tune from STUDIES + CALLS
## original
nohup python train_ms.py -c configs/studies-teacher_base_ft.json -m studies-teacher_base_ft -f studies-calls_base > logs/studies-teacher_base_ft.out
## with x-vector
nohup python train_embed.py -c configs/studies-teacher_xvector_ft.json -m studies-teacher_xvector_ft -f studies-calls_xvector > logs/studies-teacher_xvector_ft.out
## with SSL model feature
nohup python train_embed_ssl.py -c configs/studies-teacher_sslfeature_ft.json -m studies-teacher_sslfeature_ft -f studies-calls_sslfeature > logs/studies-teacher_sslfeature_ft.out
## with SER posterior probability
nohup python train_embed.py -c configs/studies-teacher_serpp_ft.json -m studies-teacher_serpp_ft -f studies-calls_serpp > logs/studies-teacher_serpp_ft.out
## with Emotion representation
nohup python train_embed.py -c configs/studies-teacher_emorepresentation_ft.json -m studies-teacher_emorepresentation_ft -f studies-calls_emorepresentation > logs/studies-teacher_emorepresentation_ft.out
## with SER posterior probability ("JTES" SER model)
nohup python train_embed.py -c configs/studies-teacher_jserpp_ft.json -m studies-teacher_jserpp_ft -f studies-calls_jserpp > logs/studies-teacher_jserpp_ft.out
## with Emotion representation ("JTES" SER model)
nohup python train_embed.py -c configs/studies-teacher_jemorepresentation_ft.json -m studies-teacher_jemorepresentation_ft -f studies-calls_jemorepresentation > logs/studies-teacher_jemorepresentation_ft.out
```

## STUDIES (three speaker model) training

This model is used for voice conversion.

```bash
# preprocess dataset
mkdir dataset
. prep_dataset/00_copy_dataset.sh /work/abelab4/s_koha/s_koha_work/dataset/STUDIES_22k STUDIES
python prep_dataset/05_prep_studies.py

# train
nohup python train_ms.py -c configs/studies_vc.json -m studies_vc > logs/studies_vc.out
```

Also, we can train another LITE-DATA model.
This model do not consisit of the student's emotional speech, only "Neutral" speech.

```bash
# preprocess dataset
python prep_dataset/06_prep_studies-lite.py

# train
nohup python train_ms.py -c configs/studies_vc-lite.json -m studies_vc-lite > logs/studies_vc-lite.out
```

## LJSpeech training

``` bash
# preprocess dataset
mkdir DUMMY1
ln -s /abelab/DB4/LJSpeech-1.1/wavs/*.wav DUMMY1/

# train
nohup python train.py -c configs/ljs_base.json -m ljs_base > logs/ljs_base.out
```

## Exporting Conda's environment

```bash
conda env export --no-build -n vits > env_conda.yml
```
