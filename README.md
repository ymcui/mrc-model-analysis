# mrc-model-analysis
This repository contains source code for our paper "Quantitative Explainability Analyses on Machine Reading Comprehension Models" (under review at journal).

## Requirements
```
Python 3.7
TensorFlow 1.15
```

All experiments are carried out using TPU. 
If you are using other training devices, please adjust these scripts accordingly.

### How to run SQuAD baseline (TPU)
Run the following script (`run.squad.sh`):
```
python -u run_squad.py \
--vocab_file=./bert/cased_L-12_H-768_A-12/vocab.txt \
--bert_config_file=./bert/cased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=./bert/cased_L-12_H-768_A-12/bert_model.ckpt \
--do_train=True \
--train_file=./squad/train-v1.1.json \
--do_predict=True \
--predict_file=./squad/dev-v1.1.json \
--train_batch_size=64 \
--predict_batch_size=32 \
--num_train_epochs=3.0 \
--max_seq_length=512 \
--doc_stride=128 \
--learning_rate=3e-5 \
--version_2_with_negative=False \
--output_dir=./path-to-output-dir \
--do_lower_case=False \
--use_tpu=True \
--tpu_name="your-tpu-name" \
--tpu_zone="your-tpu-zone"
```

- Put pre-trained BERT checkpoint in `bert` directory
- Put SQuAD train/dev files in `squad` directory
- SQuAD train/dev file: 
	- train: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	- dev: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json


### How to run CMRC 2018 baseline (TPU)
Run the following script (`run.cmrc2018.sh`):
```
python -u run_squad_fix.py \
--vocab_file=./bert/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=./bert/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=./bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
--do_train=True \
--train_file=./cmrc2018/cmrc2018_train.json \
--do_predict=True \
--predict_file=./cmrc2018/cmrc2018_dev.json \
--train_batch_size=64 \
--predict_batch_size=32 \
--num_train_epochs=2 \
--max_seq_length=512 \
--doc_stride=128 \
--learning_rate=3e-5 \
--do_lower_case=True \
--output_dir=./path-to-output-dir \
--use_tpu=True \
--tpu_name="your-tpu-name" \
--tpu_zone="your-tpu-zone"
```

- Put pre-trained BERT checkpoint in `bert` directory
- Put CMRC 2018 train/dev files in `cmrc2018` directory
- CMRC 2018 train/dev file: https://github.com/ymcui/cmrc2018/tree/master/squad-style-data


### How to mask attention zones 

Simply pass an additional argument `--mask_zone` to `run_squad.py` or `run_cmrc2018.py` script.
The followings are valid values for `--mask_zone`:
- "no_q2": masking Q2 zone
- "no_q2p": masking Q2P zone
- "no_p2": masking P2 zone
- "no_p2q": masking P2Q zone

Note: 
- This will mask specified attention zone in **ALL layers**. If you want to mask a specific layer, please modify `attention_layer()` in `modeling.py` and pass layer number to this function.
- By default, this will mask Top-10 values in that attention zone. If you wish to mask more values, please modify `mask_top_n` variable in `modeling.py -> attention_layer()`.

## Citation
```bibtex
TBA
```