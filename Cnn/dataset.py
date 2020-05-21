# -*- coding: utf-8 -*-

from torchtext.data import Field, TabularDataset
from collections import Counter
from torchtext.vocab import Vocab

import spacy

#from sklearn.model_selection import train_test_split

spacy_en = spacy.load('en')

def tokenizer(text):
    """
    Tokenizes text from a string into a list of strings
    """
    text = text.lower()
    return [tok.text for tok in spacy_en.tokenizer(text)]

def merge_vocabs(vocabs, vocab_size=None):
    """
    Merge individual vocabularies (assumed to be generated from disjoint
    documents) into a larger vocabulary.
    Args:
        vocabs: `torchtext.vocab.Vocab` vocabularies to be merged
        vocab_size: `int` the final vocabulary size. `None` for no limit.
    Return:
        `torchtext.vocab.Vocab`
    """
    merged = sum([vocab.freqs for vocab in vocabs], Counter())
    return Vocab(merged, specials=['<pad>','<unk>','<sep>','<sos>','<eos>'], vectors = 'fasttext.en.300d')

Q = Field(tokenize = tokenizer,
            #sequential =True,
            init_token = '<sos>', 
            eos_token = '<sep>',
            pad_token = '<pad>',
            unk_token = '<unk>',
            #lower = True,
            batch_first = True,
            include_lengths = True)

A = Field(tokenize = tokenizer, 
            #sequential =True,
            eos_token = '<eos>',
            pad_token = '<pad>',
            unk_token = '<unk>',
            #lower = True,
            batch_first = True,
            include_lengths = True)


Ans_Sen = Field(tokenize = tokenizer,
            #sequential =True,
            init_token = '<sos>', 
            eos_token = '<eos>', 
            pad_token = '<pad>',
            unk_token = '<unk>',
            #lower = True, 
            batch_first = True)
        
"""def extract_QnA_Ans_Sent(tab):
    Preprocessing of Data

    input_tuple =[]
    extract = []
    
    for index, item in enumerate(tab):
        q = item['question']
        a = item['answer']
        a_s = item['answer_sentence']
        input_tuple = [q,a,a_s]
        extract.append(input_tuple)
        
    return extract #[[data['question']+" <sep> "+data['answer'], data['answer_sentence']] for data in tab]


def make_torchtext(data, fields):
    examples = [Example.fromlist(i, fields) for i in data]
    return Dataset(examples, fields)
"""
def loadDataset():
    """
    Loading and Spliting Dataset
    """
    train_data, test_data = [], []
    
    print("==> Loading Training Set")
    
    fields_tuple = {"question":('Q',Q),"answer":('A',A),"answer_sentence":('Ans_Sen',Ans_Sen)}

    train_data = TabularDataset(path='data/Extended_Dataset_Train.json', format='json', fields=fields_tuple)
    print("Size of Training Set : {}".format(len(train_data)))
    print("Training Set Example: {}".format(train_data[0].__dict__))
    
    print("==> Loading Test Set")
    test_data = TabularDataset(path='data/Extended_Dataset_Test.json', format='json', fields=fields_tuple)

    #print(dataset.__dict__.keys())
    #dataset = extract_QnA_Ans_Sent(dataset)
    
    
    
    #print("==> Creating Training Set and Test Set")
    
    #train_data, test_data = dataset.split(split_ratio=0.8)
    #train, val = train_test_split(train, test_size=0.2)
    
    
    #train_data = make_torchtext(train,fields_tuple)
    #test_data = make_torchtext(test,fields_tuple)
    
    
    print("Size of Test Set : {}".format(len(test_data)))
    print("Test Set Example: {}".format(test_data[0].__dict__))

    #val_data = make_torchtext(val,fields_tuple)
    
    print("==> Building Vocabulary using Fasttext")
    
    Q.build_vocab(train_data, specials = ['<sep>'], vectors = 'fasttext.en.300d')
    A.build_vocab(train_data, vectors = 'fasttext.en.300d')
    Ans_Sen.build_vocab(train_data, vectors = 'fasttext.en.300d')
    
    QnA_vocab = merge_vocabs([Q.vocab,A.vocab,Ans_Sen.vocab])
    
    #fields = [('A',A),('Q',Q),('Ans_Sen',Ans_Sen)]
    
    return train_data, test_data, QnA_vocab #, fields
"""
train_data, test_data, QnA_vocab = loadDataset()
print(train_data)
print("Dataset Example: {}".format(train_data[0].__dict__))
print(test_data)
print("Dataset Example: {}".format(test_data[3].__dict__))
print(QnA_vocab)
"""
