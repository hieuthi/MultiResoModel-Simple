#!/bin/bash


if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi

mkdir -p downloads

echo "$0: download PartialSpoof database_eval.tar.gz."
if [ ! -f downloads/database_eval.tar.gz ]; then wget https://zenodo.org/api/records/5766198/files/database_eval.tar.gz/content -O downloads/database_eval.tar.gz; fi
echo "$0: download PartialSpoof database_train.tar.gz."
if [ ! -f downloads/database_train.tar.gz ]; then wget https://zenodo.org/api/records/5766198/files/database_train.tar.gz/content -O downloads/database_train.tar.gz; fi
echo "$0: download PartialSpoof database_dev.tar.gz."
if [ ! -f downloads/database_dev.tar.gz ]; then wget https://zenodo.org/api/records/5766198/files/database_dev.tar.gz/content -O downloads/database_dev.tar.gz; fi


echo "$0: extract data to PartialSpoof/"
mkdir -p PartialSpoof/wav
if [ ! -d PartialSpoof/wav/eval ]; then tar -xvf downloads/database_eval.tar.gz && mv downloads/database/eval/con_wav PartialSpoof/wav/eval; fi
if [ ! -d PartialSpoof/wav/train ]; then tar -xvf downloads/database_train.tar.gz && mv downloads/database/train/con_wav PartialSpoof/wav/train; fi
if [ ! -d PartialSpoof/wav/dev ]; then tar -xvf downloads/database_dev.tar.gz && mv downloads/database/dev/con_wav PartialSpoof/wav/dev; fi

echo "$0: download labels for PartialSpoof"
if [ ! -d PartialSpoof/label_PartialSpoof_dev.txt ]; then wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.1.0/label_PartialSpoof_dev.txt -P PartialSpoof; fi
if [ ! -d PartialSpoof/label_PartialSpoof_train.txt ]; then wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.1.0/label_PartialSpoof_train.txt -P PartialSpoof; fi
if [ ! -d PartialSpoof/label_PartialSpoof_eval.txt ]; then wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.1.0/label_PartialSpoof_eval.txt -P PartialSpoof; fi

echo "$0: prepare scp files"
mkdir -p PartialSpoof/scp
if [ ! -d PartialSpoof/scp/wav-train.scp ]; then find PartialSpoof/wav/train -name "*.wav" | sort > PartialSpoof/scp/wav-train.scp; fi
if [ ! -d PartialSpoof/scp/wav-dev.scp ]; then find PartialSpoof/wav/dev -name "*.wav" | sort > PartialSpoof/scp/wav-dev.scp; fi
if [ ! -d PartialSpoof/scp/wav-eval.scp ]; then find PartialSpoof/wav/eval -name "*.wav" | sort > PartialSpoof/scp/wav-eval.scp; fi


## Training detection model with PartialSpoof
echo "$0: train the multiresomodel with PartialSpoof"
python -u train.py --config configs/baseline.toml --batch_size 8 --num_workers 6 > logs/baseline.log 2>&1 &

echo "$0: the training is finished edit the script if you want to run evaluation"
# ## Evaluating with PartialSpoof eval set
# best_epoch=9
# python -u train.py --config configs/baseline.toml --batch_size 4 \ 
#   --is_eval --eval_scp PartialSpoof/scp/wav-eval.scp \ 
#   --resume_ckpt exp/baseline/${best_epoch}.pth --eval_tag "_partialspoof-eval" 2>&1 &