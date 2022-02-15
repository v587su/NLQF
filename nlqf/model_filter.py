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
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer 
from model import *



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


def _compute_loss(model, data_loader, vocab_size, device, with_cuda):
    loss_values = []
    decoded = []

    loss_weight = torch.ones((vocab_size)).to(device)
    loss_weight[0] = 0
    loss_weight[1] = 0

    loss = nn.CrossEntropyLoss(weight=loss_weight)
    for data, target in tqdm.tqdm(data_loader):
        if with_cuda:
            torch.cuda.empty_cache()
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device)
            m, l, z, decoded = model.forward(data)
            decoded = decoded.view(-1, vocab_size)
            loss_value = loss(decoded.view(-1, vocab_size), target.view(-1))
            loss_values.append(loss_value.item())
    return np.array(loss_values)


def model_filter(comments, word_vocab, model, with_cuda=True, query_len=20,num_workers=1,max_iter=1000,dividing_point=None):
    if not isinstance(comments, list):
        raise TypeError('comments must be a list')
    
    if len(comments) < 2:
        raise ValueError('The length of comments must be greater than 1')

    vocab_size = len(word_vocab)
    corpus = QueryDataset(comments, word_vocab, query_len, is_docstring=True)
    data_loader = torch.utils.data.DataLoader(
        dataset=corpus, batch_size=1, shuffle=False, num_workers=num_workers)

    model.eval()
    device = torch.device("cuda:0" if with_cuda else "cpu")
    model = model.to(device)

    losses = _compute_loss(model,data_loader, vocab_size, device, with_cuda)
    idx = np.argsort(losses)
    losses = np.array(losses)[idx].reshape(-1, 1)

    if dividing_point is None:
        gmm = GaussianMixture(
            n_components=2, covariance_type='full', max_iter=max_iter).fit(losses)
        predicted = gmm.predict(losses)
        return_type = predicted[np.argmin(losses)]
        idx_new = idx[predicted == return_type]
        idx_new.sort()
    elif isinstance(dividing_point, int):
        idx_new = idx[:int(len(idx)*dividing_point)]
    elif isinstance(dividing_point, function):
        idx_new = dividing_point(idx,losses)

    filtered_comments = [comments[i] for i in idx_new]
    return filtered_comments, idx_new
