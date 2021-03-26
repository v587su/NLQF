# Code for "On the importance of Building High-quality Training Datasets for Neural Code Search"
This repo contains the source code and dataset introduced in our paper.

## Tool: Natural Language Query Filter (NLQF)

NLQF is a tool to filter query-appropriate text for building high-quality code search datasets. It consists of rule-based filering and model-based filtering.

### Installation

```
pip install NLQF
```

### Usage
*NOTE: To use this tool, you need to download the [pre-trained model](https://github.com/v587su/comment-filter/blob/master/resource/vae.model) and the [word vocabulary](https://github.com/v587su/comment-filter/blob/master/resource/word_vocab.json).*

```
import nlqf
import torch 
import json

vocab_path = './word_vocab.json'
model_path = './vae.model'
with open(vocab_path, 'r') as f:
    word_vocab = json.load(f)
model = torch.load(model_path)

raw_comments = ['======','create a remote conection','return @test','convert int to string']

comments,idx = nlqf.rule_filter(raw_comments)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1, 3]

comments,idx = nlqf.model_filter(raw_comments, word_vocab, model)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1 3]
```

### VAE Model 
To train the filtering model with your own real query corpus, you can refer to [this repository](https://github.com/v587su/VAE_public).

## Dataset: Codebase with Filtered Comments (COFIC)
This dataset can be downloaded at [GoogleDrive](https://drive.google.com/file/d/1GILk46cxKx64EEBNVyswEjHaLHuF5V_x/view?usp=sharing) or [BaiduNetdisk](https://pan.baidu.com/s/1oKhEpo1r5XmAoKiZygxI0Q) (code: es5f).