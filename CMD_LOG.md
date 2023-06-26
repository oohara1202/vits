# Command

## Setup LJSpeech

``` bash
mkdir DUMMY1
ln -s /abelab/DB4/LJSpeech-1.1/wavs/*.wav DUMMY1/
```

## Training LJSpeech

```bash
python train.py -c configs/ljs_base.json -m ljs_base
```
