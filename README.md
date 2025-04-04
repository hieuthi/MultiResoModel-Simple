# MultiResoModel (Simple)

This repository is an unofficial reimplementation of the [Partial Spoof Detection MultiResoModel](https://ieeexplore.ieee.org/document/10003971) that was used in the paper [LlamaPartialSpoof](https://arxiv.org/abs/2409.14743). It was completely rewritten to include only the most essential parts and to make reproduction and improvement simple.

## Notice
This model configuration is not an exactly replicate of [the original MultiResoModel](https://github.com/nii-yamagishilab/PartialSpoof/tree/main/03multireso) hence the slight different result. Some of the major difference are.
- This model was trained on fixed-length segments by clipping or padding instead of entire utterance like the original to make it's easier to adapt to different dataset.
- Model is trained on a random segment of utterance instead of from the start

## How to Use
### Training
- For a start to finish training please execute
```
./train_baseline.sh
```
- To customize the training process, please edit the config file. For examples: changing the training and validation dataset. I used a custom label format for partial spoof dataset.

### Inference
- For inference, you need a checkpoint either from the training or download our [checkpoints](#Checkpoint)
```bash
./inference.sh baseline 55 PartialSpoof/wav/eval ps-eval
```
- Since the model was trained on fixed-segment, the inference will first split the evaluation data into multiple fixed-length segments before running the inference then combined the results.

### Evaluation
```
./evaluate.sh baseline 55 ps-eval
```

## Checkpoints
- You can download checkpoints from [huggingface](https://huggingface.co/hieuthi/MultiResoModel-Simple-ckpts). Note that the checkpoints on huggingface are different runs from the original LlamaPartialSpoof paper hence the slightly different results.
- Utterance-based Equal Error Rate (EER)

|           Model | ps-eval |
|-----------------|---------|
| baseline-ps-e55 |   1.48% |

- 20-ms frame-based EER

|           Model | ps-eval |
|-----------------|---------|
| baseline-ps-e55 |  13.67% |

## Citations
If using this source code please cite both the paper introduced this reimplementation and the original paper
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
