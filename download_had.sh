#!/bin/bash


if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi

mkdir -p downloads

echo "$0: download LlamaPartialSpoof R01TTS.0.a.tgz."
if [ ! -f downloads/HAD.zip ]; then wget https://zenodo.org/api/records/10377492/files/HAD.zip/content -O downloads/HAD.zip; fi


if [ ! -d HAD/ ]; then unzip downloads/HAD.zip; fi

mkdir -p HAD/wav;
if [ ! -d HAD/wav/test/ ]; then unzip HAD/HAD_test/test.zip -d HAD/wav; fi

echo "$0: prepare scp files"
mkdir -p HAD/scp
if [ ! -f HAD/scp/test.scp ]; then find HAD/wav/test -name "*.wav" | sort > HAD/scp/test.scp; fi

if [ ! -f HAD/label_HAD_dev.txt ]; then wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.2.0/label_HAD_dev.txt -P HAD; fi
if [ ! -f HAD/label_HAD_train.txt ]; then wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.2.0/label_HAD_train.txt -P HAD; fi
if [ ! -f HAD/label_HAD_test.txt ]; then wget https://github.com/hieuthi/MultiResoModel-Simple/releases/download/v0.2.0/label_HAD_test.txt -P HAD; fi
