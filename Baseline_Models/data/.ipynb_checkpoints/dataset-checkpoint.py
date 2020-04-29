from torchtext.data import Field, Dataset, Example

import spacy
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split

spacy_en = spacy.load('en')

def tokenizer(text):
    """
    Tokenizes text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

QnA = Field(tokenize = tokenizer, 
            init_token = '<sos>', 
            eos_token = '<eos>',
            lower = True,
            batch_first = True,
            include_lengths = True)

Ans_Sen = Field(tokenize = tokenizer, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)
        
def extract_QnA_Ans_Sent(tab):
    """
    Preprocessing of Data
    """
    input_tuple =[]
    extract = []
    
    for index, item in enumerate(tab):
        src = item['question']+ " <sep> "+ item['answer']
        trg = item['answer_sentence']
        input_tuple = [src,trg]
        extract.append(input_tuple)
        
    return extract #[[data['question']+" <sep> "+data['answer'], data['answer_sentence']] for data in tab]


def make_torchtext(data, fields):
    examples = [Example.fromlist(i, fields) for i in data]
    return Dataset(examples, fields)

def loadDataset():
    """
    Loading and Spliting Dataset
    """
    train, test, val = [], [], []
    
    print("==> Loading Training Set")
    
    with open("./data/Batch_1_train_filter.json") as json_file:
        train = json.load(json_file)
        #train = pd.DataFrame(train_file)
        
    print("==> Loading Testing Set")
    
    with open("./data/Batch_1_test_filter.json") as json_file:
        test = json.load(json_file)
        #test = pd.DataFrame(test_file)
    
    print("==> Preprocessing Data")
    
    train = extract_QnA_Ans_Sent(train)
    test = extract_QnA_Ans_Sent(test)
    
    print("==> Creating Validation Set from Training Set")
    
    train, val = train_test_split(train, test_size=0.2, shuffle=False)
    
    fields_tuple = [('QnA',QnA),('Ans_Sen',Ans_Sen)]
    
    train_data = make_torchtext(train,fields_tuple)
    test_data = make_torchtext(test,fields_tuple)
    val_data = make_torchtext(val,fields_tuple)
    
    print("==> Building Vocabulary using Glove")
    
    QnA.build_vocab(train_data, vectors = 'glove.6B.100d')
    Ans_Sen.build_vocab(train_data, vectors = 'glove.6B.100d')
    
    return train_data, test_data, val_data, QnA.vocab, Ans_Sen.vocab




    
