# MultiResoModel (Simple)

This repository is an unofficial reimplementation of the [Partial Spoof Detection MultiResoModel](https://ieeexplore.ieee.org/document/10003971) that was used in the paper [LlamaPartialSpoof](https://arxiv.org/abs/2409.14743). It was completely rewritten to include only the most essential parts and to make reproduction and improvement simple.

## Notice
This is not an exact replication of [the original MultiResoModel](https://github.com/nii-yamagishilab/PartialSpoof/tree/main/03multireso) hence the slight different result. Some of the major differences are:
- This model was trained on fixed-length segments instead of entire utterance to easier adapted to different dataset.
- Model is trained on a random segment of utterance instead of from the start

## How to Use
### Training
- For a start to finish training execute
```
./train_baseline.sh
```
- To customize the training process you need to edit the config file. For examples: changing the training and validation dataset. I used a custom label format for partial spoof dataset.

### Inference
- For inference, you need a checkpoint either from the training or download our [checkpoints](#Checkpoint)
```bash
./inference.sh baseline 55 PartialSpoof/wav/eval ps-eval
```
- Since the model was trained on fixed-segment, the inference will first split the evaluation data into multiple fixed-length segments before running the inference then combined the results.

### Evaluation
```
./evaluate.sh baseline 55 PartialSpoof/label_PartialSpoof_eval.txt ps-eval
```

### Out-of-domain Evaluation
#### LlamaPartialSpoof
- LlamaPartialSpoof was prepared as an out-of-domain evaluation dataset. Similar to in-domain evaluation you can get the result by running the following commands
```
# Download LlamaPartialSpoof
./download_lps.sh

# Inference
./inference.sh baseline 55 LlamaPartialSpoof/R01TTS.0.a lps0a

# Evaluation
./evaluate.sh baseline 55 LlamaPartialSpoof/label_R01TTS.0.a.txt lps0a

```
- The evaluation script calculates base on the utterances included in the label file. You can get full fakes and partial fakes only results using the follow scripts
```
./evaluate.sh baseline 55 LlamaPartialSpoof/extras/label_bonafide_full.txt lps0a "_full"
./evaluate.sh baseline 55 LlamaPartialSpoof/extras/label_bonafide_partial.txt lps0a "_partial"

```

#### Half-truth
- Similarly for Half-truth data
```
./download_had.sh
./inference.sh baseline 55 HAD/wav/test had-test
./evaluate.sh baseline 55 HAD/label_HAD_test.txt had-test
```


## Checkpoints
- You can download checkpoints from [huggingface](https://huggingface.co/hieuthi/MultiResoModel-Simple-ckpts). Note that the checkpoints on huggingface are different runs from the original LlamaPartialSpoof paper hence the slightly different results.

- Utterance-based Equal Error Rate (EER)

|           Model | ps-eval | LlamaPartialSpoof |
|-----------------|--------:|------------------:|
| baseline-ps-e55 |   1.48% |            24.51% |

- 20-ms frame-based EER

|           Model | ps-eval | LlamaPartialSpoof | Half-truth Test |
|-----------------|---------|------------------:|----------------:|
| baseline-ps-e55 |  13.67% |           46.30%  |          46.48% |

## Citations
If using this source code please cite both the LlamaPartialSpoof paper which introduced this reimplementation and the original MultiResoModel paper
- LlamaPartialSpoof
```
@inproceedings{luong2025llamapartialspoof,
  title={LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation},
  author={Luong, Hieu-Thi and Li, Haoyang and Zhang, Lin and Lee, Kong Aik and Chng, Eng Siong},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

- MultiResoModel
```
@article{10003971,
  author={Zhang, Lin and Wang, Xin and Cooper, Erica and Evans, Nicholas and Yamagishi, Junichi},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance}, 
  year={2023},
  volume={31},
  number={},
  pages={813-825},
  doi={10.1109/TASLP.2022.3233236}}
```

## License

[MIT License](LICENSE)

Please note that certain code adaptations from external repositories may be subject to alternative licensing terms, as specified in the applicable subdirectory.
