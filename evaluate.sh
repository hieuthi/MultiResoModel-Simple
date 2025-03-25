#!/bin/bash

model=$1
epoch=$2
datadir=$3

tag=$4
if [ -z "$tag" ]; then tag=$( basename $datadir ); fi

MAXDUR=9.6

# Check if the epoch exists
checkpoint=exp/${model}/${epoch}.pth
if [ ! -f ${checkpoint} ]; then
  echo "ERROR: ${checkpoint} does not exist"
  exit 1;
fi

# Cut long audio to max length
normdir=${datadir%/}_lt${MAXDUR}
if [ ! -d $normdir ]; then
  mkdir -p $normdir
  while read line; do
    dur=$( soxi -D $line )
    echo $line $dur
    if (( $(echo "$dur > $MAXDUR" | bc -l) )); then
      bname=$( basename $line .wav )
      sox $line $normdir/${bname}_PART.wav trim 0 $MAXDUR : newfile : restart
    else
      cp $line $normdir
    fi
  done < <( find ${datadir}/ -name "*.wav" )
fi

# Prepare script file
outdir=exp/$model/result_${model}_e${epoch}_${tag}
mkdir -p $outdir
find $normdir/ -name "*.wav" > exp/$model/${tag}_lt${MAXDUR}.scp

# Inference
python -u train.py --config configs/${model}.toml --batch_size 4 --is_eval --eval_scp exp/$model/${tag}_lt${MAXDUR}.scp --resume_ckpt ${checkpoint} --eval_tag "_${tag}"
