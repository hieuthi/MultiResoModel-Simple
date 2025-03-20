# MultiResoModel (Simple)

This repository is an unofficial reimplementation of the [Partial Spoof Detection MultiResoModel](https://ieeexplore.ieee.org/document/10003971) to conduct experiments with the LlamaPartialSpoof Dataset. It was completely rewritten to include only the most essential parts of the model and to make reproduction and improvement simple.

This repository is still underdevelopment. More information will be added soon.

## How to Use
### Training
- Training looks something like this
```bash
python -u train.py --config configs/baseline.toml --batch_size 8 --num_workers 6 > logs/baseline.log 2>&1 &
```
- The information about the training and developing data is specifed in the config files, so you need to edit it to your local environment.
- The `scp_train_wav` config is the text file with a list of audio file. The `scp_train_lab` config is a specific label format for partial spoof dataset. It is the same format as [LlamaPartialSpoof Dataset](https://zenodo.org/records/14214149). I will upload a pre-formatted labels for third party datasets in the future. The `scp_train_wav` and the `scp_train_lab` must have the same order.

### Evaluating
- Evaluating looks something like this
```bash
python -u train.py --config configs/baseline.toml --batch_size 4 --is_eval --eval_scp PartialSpoof/scp/wav-eval.scp --resume_ckpt exp/baseline/9.pth --eval_tag "_partialspoof-eval" 2>&1 &
```
- You need to choose a checkpoint by yourself. The script will produce multiple text files that store the results in multiple resolutions at `exp/baseline/result_baseline_e9_partialspoof-eval`


## Citations
If you use this source code for your research please cite both the original paper and the paper introduce this version

- LlamaPartialSpoof Dataset
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