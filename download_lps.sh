#!/bin/bash


if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi

mkdir -p downloads

echo "$0: download LlamaPartialSpoof R01TTS.0.a.tgz."
if [ ! -f downloads/R01TTS.0.a.tgz ]; then wget https://zenodo.org/api/records/14214149/files/R01TTS.0.a.tgz/content -O downloads/R01TTS.0.a.tgz; fi

echo "$0: extract data to LlamaPartialSpoof/"
mkdir -p LlamaPartialSpoof
if [ ! -d LlamaPartialSpoof/R01TTs.0.a ]; then tar -xvf downloads/R01TTS.0.a.tgz -C downloads/ && mv downloads/R01TTS.0.a LlamaPartialSpoof; fi
if [ ! -f LlamaPartialSpoof/label_R01TTS.0.a.txt ]; then wget https://zenodo.org/api/records/14214149/files/label_R01TTS.0.a.txt/content -O LlamaPartialSpoof/label_R01TTS.0.a.txt; fi

echo "$0: prepare scp files"
mkdir -p LlamaPartialSpoof/scp
if [ ! -f LlamaPartialSpoof/scp/R01TTS.0.a.scp ]; then find LlamaPartialSpoof/R01TTS.0.a -name "*.wav" | sort > LlamaPartialSpoof/scp/R01TTS.0.a.scp; fi

mkdir -p LlamaPartialSpoof/extras
grep dev-clean LlamaPartialSpoof/label_R01TTS.0.a.txt > LlamaPartialSpoof/extras/label_bonafide.txt
grep test-clean LlamaPartialSpoof/label_R01TTS.0.a.txt >> LlamaPartialSpoof/extras/label_bonafide.txt

cp LlamaPartialSpoof/extras/label_bonafide{,_full}.txt 
cp LlamaPartialSpoof/extras/label_bonafide{,_partial}.txt 

grep full LlamaPartialSpoof/label_R01TTS.0.a.txt >> LlamaPartialSpoof/extras/label_bonafide_full.txt
grep partial LlamaPartialSpoof/label_R01TTS.0.a.txt >> LlamaPartialSpoof/extras/label_bonafide_partial.txt
