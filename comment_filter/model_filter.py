import json
import torch
import tqdm
import numpy as np
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(__file__))
from torch.utils.data import Dataset
from sklearn.mixture import GaussianMixture
from nltk.stem import WordNetLemmatizer 
from model import *

default_options = {
    'with_cuda': True,
    'query_len': 20,
    'num_workers': 1,
    'max_iter':1000
}


class QueryDataset(Dataset):
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    def __init__(self, query, word_vocab, query_len, is_docstring=False):
        super(QueryDataset, self).__init__()
        self.query_len = query_len
        self.word_vocab = word_vocab
        self.query = []
        lemmatizer = WordNetLemmatizer()  
        
        if isinstance(query,str):
            with open(query, 'r') as f:
                lines = f.readlines()
        else:
            lines = query
            
        for q in lines:
            words = q.lower().split()[:query_len]
            if is_docstring:
                if len(words)>0 and word_vocab.get(words[0],self.UNK) == self.UNK:
                    words[0] = lemmatizer.lemmatize(words[0],pos='v')
                words = [word_vocab.get(w,self.UNK) for w in words]+[self.EOS]
            else:
                words = [int(w) for w in words]+[self.EOS]
            padding = [self.PAD for _ in range(query_len - len(words)+2)]
            words.extend(padding)
            self.query.append(words)


    def __len__(self):
        return len(self.query)

    def __getitem__(self, item):
        return torch.tensor(self.query[item]),torch.tensor(self.query[item])


def _compute_loss(opt, model, data_loader, vocab_size, device):
    loss_values = []
    decoded = []

    loss_weight = torch.ones((vocab_size)).to(device)
    loss_weight[0] = 0
    loss_weight[1] = 0

    loss = nn.CrossEntropyLoss(weight=loss_weight)
    for data, target in tqdm.tqdm(data_loader):
        if opt['with_cuda']:
            torch.cuda.empty_cache()
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            m, l, z, decoded = model.forward(data)
            decoded = decoded.view(-1, vocab_size)
            loss_value = loss(decoded.view(-1, vocab_size), target.view(-1))
            loss_values.append(loss_value.item())
    return np.array(loss_values)


def model_filter(comments, vocab_path='comment_filter/data/word_vocab.json', model_path='comment_filter/data/vae.model', options={}):
    if not isinstance(comments, list):
        raise TypeError

    default_options.update(options)

    with open(vocab_path, 'r') as f:
        word_vocab = json.load(f)
    vocab_size = len(word_vocab)

    corpus = QueryDataset(comments, word_vocab,
                          default_options['query_len'], is_docstring=True)
    data_loader = torch.utils.data.DataLoader(
        dataset=corpus, batch_size=1, shuffle=False, num_workers=default_options['num_workers'])

    model = torch.load(model_path)
    model.eval()
    device = torch.device("cuda:0" if default_options['with_cuda'] else "cpu")
    model = model.to(device)

    losses = _compute_loss(default_options, model,
                           data_loader, vocab_size, device)
    idx = np.argsort(losses)
    losses = np.array(losses)[idx].reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=2, covariance_type='full', max_iter=default_options['max_iter']).fit(losses)
    predicted = gmm.predict(losses)
    return_type = predicted[np.argmin(losses)]
    idx_new = idx[predicted == return_type]
    idx_new.sort()

    filtered_comments = [comments[i] for i in idx_new]
    return filtered_comments, idx_new
