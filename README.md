## COMMENT FILTER

Comment Filter is a tool to filter query-appropriate comments for building high-quality code search datasets. It consists of rule-based filering and model-based filtering.

### Installation

```
pip install comment_filter
```

### Usage
*NOTE: To use this tool, you need to download the [pre-trained model](https://github.com/v587su/comment-filter/blob/master/resource/vae.model) and the [word vocabulary](https://github.com/v587su/comment-filter/blob/master/resource/word_vocab.json).*

```
import comment_filter
import torch 
import json

vocab_path = './word_vocab.json'
model_path = './vae.model'
with open(vocab_path, 'r') as f:
    word_vocab = json.load(f)
model = torch.load(model_path)

raw_comments = ['======','create a remote conection','return @test','convert int to string']

comments,idx = comment_filter.rule_filter(raw_comments)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1, 3]

comments,idx = comment_filter.model_filter(raw_comments, word_vocab, model)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1 3]
```