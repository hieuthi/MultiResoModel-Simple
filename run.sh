#!/bin/bash


if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi

mkdir -p downloads

data_url=https://zenodo.org/api/records/5766198/files/database_eval.tar.gz/content
if [ ! -f downloads/database_eval.tar.gz ]; then
  echo "$0: downloading data PartialSpoof database_eval.tar.gz."
  wget $data_url -O downloads/database_eval.tar.gz
fi

data_url=https://zenodo.org/api/records/5766198/files/database_train.tar.gz/content
if [ ! -f downloads/database_train.tar.gz ]; then
  echo "$0: downloading data PartialSpoof database_train.tar.gz."
  wget $data_url -O downloads/database_train.tar.gz
fi

data_url=https://zenodo.org/api/records/5766198/files/database_dev.tar.gz/content
if [ ! -f downloads/database_dev.tar.gz ]; then
  echo "$0: downloading data PartialSpoof database_dev.tar.gz."
  wget $data_url -O downloads/database_dev.tar.gz
fi


echo "$0: extract data to PartialSpoof"
mkdir -p PartialSpoof/wav
tar -xvf downloads/database_eval.tar.gz && mv downloads/database/eval/con_wav PartialSpoof/wav/eval
tar -xvf downloads/database_train.tar.gz && mv downloads/database/train/con_wav PartialSpoof/wav/train
tar -xvf downloads/database_dev.tar.gz && mv downloads/database/dev/con_wav PartialSpoof/wav/dev

