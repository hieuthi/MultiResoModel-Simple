#!/bin/bash


if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi

mkdir -p downloads

if [ ! -f downloads/database_eval.tar.gz ]; then
  echo "$0: downloading data PartialSpoof database_eval.tar.gz."
  wget https://zenodo.org/api/records/5766198/files/database_eval.tar.gz/content -O downloads/database_eval.tar.gz
fi

if [ ! -f downloads/database_train.tar.gz ]; then
  echo "$0: downloading data PartialSpoof database_train.tar.gz."
  wget https://zenodo.org/api/records/5766198/files/database_train.tar.gz/content -O downloads/database_train.tar.gz
fi

if [ ! -f downloads/database_dev.tar.gz ]; then
  echo "$0: downloading data PartialSpoof database_dev.tar.gz."
  wget https://zenodo.org/api/records/5766198/files/database_dev.tar.gz/content -O downloads/database_dev.tar.gz
fi


echo "$0: extract data to PartialSpoof"
mkdir -p PartialSpoof/wav
tar -xvf downloads/database_eval.tar.gz && mv downloads/database/eval/con_wav PartialSpoof/wav/eval
tar -xvf downloads/database_train.tar.gz && mv downloads/database/train/con_wav PartialSpoof/wav/train
tar -xvf downloads/database_dev.tar.gz && mv downloads/database/dev/con_wav PartialSpoof/wav/dev

wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.1.0/label_PartialSpoof_dev.txt -P PartialSpoof
wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.1.0/label_PartialSpoof_train.txt -P PartialSpoof
wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.1.0/label_PartialSpoof_eval.txt -P PartialSpoof

mkdir -p PartialSpoof/scp
find PartialSpoof/wav/train -name "*.wav" | sort > scp/wav-train.scp
find PartialSpoof/wav/dev -name "*.wav" | sort > scp/wav-dev.scp
find PartialSpoof/wav/eval -name "*.wav" | sort > scp/wav-eval.scp