# CWS

中文分词

# Requirements

* python: 3.7.1
* pytorch: 1.0.1

# Usage

## 环境

```sh
$ pip install -r requirements.txt
```

## 运行

模型通过Multi-Task Learning (MTL)的方式训练ctb和pku两个规范的预料，分别为12w句和120w句  

两个语料通过corpus-weighting的方式轮流迭代，batch_size大小之比为1:10

根据是否使用了BERT，模型放在了两个不同的分支下：

```sh
# 切换到没有使用BERT的分支
$ git checkout mtl
# 切换到使用BERT的分支
$ git checkout mtl-bert
```

模型的超参放在`config.ini`这一配置文件中

注意因为BERT的限制，中文句长不能超过512（去掉句首`[CLS]`和句尾`[SEP]`两个特殊字符就是510），配置文件中统一设置最大句长为500，不使用BERT这个限制可以去掉

如果需要使用新的语料重新训练，只需修改配置文件的`[Train]`section下的路径即可

模型的训练和测试方式如下：

```sh
# device: GPU设备号，可以不指定
# config: 超参文件

# Preprocess
$ nohup python -u main.py --device=0 --config=config.ini --is-train > results/log.create-dict 2>&1 &

# Train
$ nohup python -u main.py --device=0 --config=config.ini --is-dictionary-exist --is-train > results/log.mtl.bert 2>&1 &

# Test
$ nohup python -u main.py --device=0 --config=config.ini --is-test --model-eval-num $num > log.test-$num 2>&1
```