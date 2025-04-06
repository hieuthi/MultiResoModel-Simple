#!/bin/bash


model=$1
epoch=$2
labelfile=$3
tag=$4


resultdir="result_${model}_e${epoch}_${tag}"

mkdir -p results/${resultdir}_{utt,unit0.02}

echo "Calculate utterance-based EER"
python partialspoof-metrics/calculate_eer.py --labpath ${labelfile} \
                                    --scopath exp/${model}/${resultdir}/utt.score \
                                    --savepath results/${resultdir}_utt \
                                    --scoreindex 2

echo "Calculate 20ms frame-based EER"
python partialspoof-metrics/calculate_eer.py --labpath ${labelfile} \
                                    --scopath exp/${model}/${resultdir}/unit0.02.score \
                                    --savepath results/${resultdir}_unit0.02 \
                                    --unit 0.02 \
                                    --scoreindex 3


