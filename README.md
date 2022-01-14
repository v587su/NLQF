# Code for "On the importance of Building High-quality Training Datasets for Neural Code Search"
This repo contains the source code introduced in our paper. The COFIC dataset can be downloaded [here](https://drive.google.com/file/d/1Ai0WMYrIGQQLqBC180mzUVDSbpkgO6uD/view?usp=sharing).

## Natural Language Query Filter (NLQF)

``NLQF`` is a tool to filter query-appropriate text for building high-quality code search datasets. It consists of rule-based filering and model-based filtering.

### Installation
You can install ``NLQF`` by running the following command:
```
pip install nlqf
```
The dependencies of ``NLQF`` are:
```
sklearn
nltk
numpy
pytorch
```
### Usage
Before the usage, you need to download the pre-trained model ``./resource/vae.model`` and the word vocabulary ``./resource/word_vocab.json``.
Here is the example script to use ``nlqf``:
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

# Modify this list to config your ruleset.
rule_list = ['contain_any_url','contain_any_javadoc_tag', 'contain_any_non_English','not_contain_any_letter', 'less_than_three_words', 'end_with_question_mark', 'contain_html_tag', 'remove_brackets', 'remove_html_tag']

# Define your own rule. Return False if you want to remove the comment_str. Return String if you want to modify the comment_str.
def my_rule(comment_str):
    return len(comment_str) < 10

# If the key starts with 'remove', the comment will be modified. 
my_rule_dict = {
    'length_less_than_10':my_rule
}


comments,idx = nlqf.rule_filter(raw_comments,rule_set=rule_list+list(my_rule_dict.keys()),rule_dict=my_rule_dict)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1, 3]

comments,idx = nlqf.model_filter(raw_comments, word_vocab, model, with_cuda=True, query_len=20,num_workers=1, max_iter=1000, dividing_proportion=0)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1 3]
```

### VAE Model 
To train the filtering model with your own query corpus, you can refer to [this repo](https://github.com/v587su/VAE_public) to train your own VAE model.

