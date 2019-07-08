# CWS

基于BERT的中文分词

# Requirements

* python: 3.7.1
* pytorch: 1.0.1
* pytorch_pretrained_bert: 0.6.2

# Usage

## 环境

```sh
$ pip install -r requirements.txt
```

## 运行

```sh
# device: GPU设备号，可以多GPU，可以不指定
# config: 超参文件

# Preprocess
$ nohup python -u main.py --device=0,1,2,3 --config=config.ini --is-train > results/log.create-dict 2>&1 &

# Train
$ nohup python -u main.py --device=0,1,2,3 --config=config.ini --is-dictionary-exist --is-train > results/log.mtl.bert 2>&1 &

# Test
$ nohup python -u main.py --device=0,1,2,3 --config=config.ini --is-test --model-eval-num $num > log.test-$num 2>&1
```