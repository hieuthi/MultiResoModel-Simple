#!/bin/bash

if [ ! -d PartialSpoof ]; then
  ./download_ps.sh
fi

echo "$0: download Wav2Vec2 pretrained model"
if [ ! -f pretrained/w2v_large_lv_fsh_swbd_cv_fixed.pt ]; then
  if [ ! -f pretrained/w2v_large_lv_fsh_swbd_cv.pt ]; then 
    wget --show-progress https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt -P pretrained;
  fi
  python fix_ssl.py;
fi

## Training detection model with PartialSpoof
echo "$0: train the multiresomodel with PartialSpoof"
python -u train.py --config configs/baseline.toml --batch_size 8 --num_workers 6 > logs/baseline.log 2>&1
echo "$0: the training is finished, please use inference.sh for inference"
