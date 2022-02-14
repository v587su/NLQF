# Code for "On the importance of Building High-quality Training Datasets for Neural Code Search"
This repo contains the source code introduced in our paper. The COFIC dataset can be downloaded [here](https://drive.google.com/file/d/1Ai0WMYrIGQQLqBC180mzUVDSbpkgO6uD/view?usp=sharing).

# Natural Language Query Filter (NLQF)

``NLQF`` is a Python3 library for filtering query-appropriate text to build high-quality code search datasets.
It consists of two filters: a rule-based filer and a model-based filter (Pytorch required).
Currently, ``NLQF`` should be used with GPU-enabled Pytorch.

## Installation
We have uploaded ``NLQF`` to Python Package Index (PyPi) and you can install it by the following command:
```
pip install nlqf
```

The source code of ``NLQF`` is available at https://github.com/v587su/NLQF.
You can also download the source code from the above link and install it directly with the following command:

```
pip install -r requirements.txt
python setup.py install
```

Besides, ``nlqf`` should be used with GPU-enabled Pytorch. You can refer to https://pytorch.org/get-started/locally/ to install Pytorch. Pytorch 1.3 is recommended.

## Usage
There are two functional APIs in this library: ``nlqf.rule_filter`` and ``nlqf.model_filter``.
The input of each filter is a list of comment strings and the output is a list of retained comments and a list of the index of the retained comments.

It is noteworthy that, to use the ``model_filter``, you need to download the pre-trained model ``./resource/vae.model`` and the word vocabulary ``./resource/word_vocab.json`` form https://github.com/v587su/NLQF.

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

# Select the rules to be applied from the following default ruleset:
# 'contain_any_url': remove comments containing any url
# 'contain_any_javadoc_tag': remove comments containing any javadoc tag
# 'contain_any_non_English': remove comments containing any non-English words
# 'not_contain_any_letter': remove comments not containing any letter
# 'less_than_three_words': remove comments with less than three words
# 'end_with_question_mark': remove comments ending with question mark
# 'contain_html_tag': remove comments containing html tag
# 'detach_brackets': detach the content in brackets from the comment
# 'detach_html_tag': detach the content in html tag from the comment
rule_list = ['contain_any_javadoc_tag','not_contain_any_letter']

# Define your own rule using functions. 
# Return False if you want to discard the comment string. 
# Return True if you want to retain the comment string. 
# Return String if you want to replace the comment string with your String.
def my_rule1(comment_str):
    return len(comment_str) < 10

def my_rule2(comment_str):
    if comment_str.startswith('Return'):
        return comment_str.replace('Return', '')
    else:
        return True

# The key should start with 'detach' if your rule is for detachable content.
# Otherwise, name the key as you like. 
my_rule_dict = {
    'length_less_than_10':my_rule1,
    'detach_return':my_rule2
}

comments,idx = nlqf.rule_filter(raw_comments, \
        selected_rules=rule_list,defined_rule_dict=my_rule_dict)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1, 3]

comments,idx = nlqf.model_filter(raw_comments, word_vocab, model, \
        with_cuda=True, query_len=20,num_workers=1, max_iter=1000)
print(comments,idx)
# ['create a remote conection', 'convert int to string'] [1 3]
```

## Process to manually inspect the filtering effects of NLQF
We provide a script and a small dataset ``./resource/samples.txt`` to validate the executability of NLQF.
The users can intuitively observe the cleaning effects from the printed outputs.
The scritpts are as follows (don't forget to modify the path of the required files):
```
import nlqf
import torch 
import json

vocab_path = './word_vocab.json'
model_path = './vae.model'
sample_path = './samples.txt'
with open(vocab_path, 'r') as f:
    word_vocab = json.load(f)
model = torch.load(model_path)

with open(sample_path, 'r') as f:
    raw_comments = f.readlines()

comments,idx = nlqf.rule_filter(raw_comments)
discarded_idx = set(range(len(raw_comments)))-set(idx)
print('comments discarded by rule filter:',[raw_comments[i] for i in discarded_idx])

queries,idx = nlqf.model_filter(comments, word_vocab, model, \
        with_cuda=True, query_len=20,num_workers=1, max_iter=1000)
discarded_idx = set(range(len(comments)))-set(idx)
print('comments discarded by model filter:',[comments[i] for i in discarded_idx])
print('comments finally retained:', queries)
```


## Process to reproduce the experiment in our paper using NLQF
In our paper, we train a new code search model on the filtered dataset and evaluate the performance of the trained model.
It is noteworthy that the running time is computed with GPU enabled.

### Step 1: Download the evaluation scripts and dataset.
We provide our evaluation scripts and dataset, which can be downloaded at https://drive.google.com/file/d/1Nv86JmW7fknQQQviV2agSw6XPhCCCo2S/view?usp=sharing

Decompress the downloaded project:
```
tar -zvxf icse_evaluation.tar.gz
```

The first-level structure of the decompressed project is as follows:
```
- models                # architecture of DeepCS     
- output                # directory to store the trained model
- processed_dataset     # directory to store the filtered and processed dataset
- raw_dataset           # directory to store the CodeSearch dataset
- resource              # directory to store the vocab and model for NLQF
- configs.py            # the configuration file for DeepCS
- data_loader.py        # the data loader for DeepCS
- filtering.py          # our data filtering and processing script for DeepCS
- modules.py            # the modules for DeepCS
- repr_code.py          # the script to evaluate DeepCS model
- train.py              # the script to train DeepCS model
- utils.py              # utility functions for DeepCS
- requirements.txt      # the requirements of this project
```
The project requires Python3.6.9 and the following packages:
```
jsonlines
pandas
nlqf==0.1.12
javalang==0.12.0
nltk
torch==1.3.1
numpy
tqdm
tables==3.4.3
```
You can install them by the following command:
```
cd ICSE_deep_code_search
pip install -r requirements.txt
```



### Step 2: Filter the dataset and process the filtered dataset into a format that can be used by DeepCS.
Before the filtering, you should download the code stopwords (https://github.com/BASE-LAB-SJTU/CSES_IR/blob/master/src/test/resources/stopwords/codeStopWord) and english stopwords documents (https://github.com/BASE-LAB-SJTU/CSES_IR/blob/master/src/test/resources/stopwords/codeQueryStopWord) and put them in ``~/nltk_data/corpora/stopwords``. (This path depends on where nltk is located.).

After that, run the following command (requires 4 hours to run):

```
python filtering.py
```

The filtered and processed dataset is saved in ``./processed_dataset/``.
We have pre-run this command and generated a processed dataset in this directory.
You can jump to Step 3 to save your time.

### Step 3: Train the code search model on the filtered dataset.
Run the following command to train the model (requires 4 hours to run):

```
python train.py
```

The trained model is saved in ``./output/``.
We have pre-run this command and obtained a trained model ``./output/epo100.h5``.
You can jump to Step 4 to save your time.

### Step 4: Evaluate the performance of the trained model.
Run the following command to evaluate the trained model (requires 5 minutes to run):
```
python repr_code.py --reload_from 100
```

The evaluation results will be printed in the terminal. 
It should be 0.541 MRR and 344 Hit@10.


### VAE Model 
To train the filtering model with your own query corpus, you can refer to [this repo](https://github.com/v587su/VAE_public).

