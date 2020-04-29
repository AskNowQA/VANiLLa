# -*- coding: utf-8 -*-

from torchtext.data import Field, Dataset, Example
from collections import Counter
from torchtext.vocab import Vocab

import spacy
import json

from sklearn.model_selection import train_test_split

spacy_en = spacy.load('en')

def tokenizer(text):
    """
    Tokenizes text from a string into a list of strings
    """
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
    return Vocab(merged, specials=['<pad>','<unk>','<sep>','<sos>','<eos>'], vectors = 'fasttext.simple.300d')

Q = Field(tokenize = tokenizer, 
            init_token = '<sos>', 
            pad_token = '<pad>',
            unk_token = '<unk>',
            lower = True,
            batch_first = True,
            include_lengths = True)

A = Field(tokenize = tokenizer, 
            pad_token = '<pad>',
            unk_token = '<unk>',
            lower = True,
            batch_first = True,
            include_lengths = True)


Ans_Sen = Field(tokenize = tokenizer, 
            init_token = '<sos>', 
            pad_token = '<pad>',
            unk_token = '<unk>',
            lower = True,
            batch_first = True)

Q_Pos_Tag = Field(init_token = '<sep>', eos_token = '<sep>',
               pad_token = '<pad>',
               unk_token = '<unk>', 
               batch_first = True,
               include_lengths = True)

A_Pos_Tag = Field(init_token = '<sep>', eos_token = '<eos>',
               pad_token = '<pad>',
               unk_token = '<unk>', 
               batch_first = True,
               include_lengths = True)
        
def extract_QnA_Ans_Sent(tab):
    """
    Preprocessing of Data
    """
    input_tuple =[]
    extract = []
    
    for index, item in enumerate(tab):
        q = item['question']
        a = item['answer']
        a_s = item['answer_sentence']
        q_pos = [token.tag_ for token in spacy_en(q)]
        a_pos = [token.tag_ for token in spacy_en(a)]
        a_s_pos = [token.tag_ for token in spacy_en(a_s)]
        input_tuple = [q,q_pos,a,a_pos,a_s,a_s_pos]
        extract.append(input_tuple)
        
    return extract #[[data['question']+" <sep> "+data['answer'], data['answer_sentence']] for data in tab]


def make_torchtext(data, fields):
    examples = [Example.fromlist(i, fields) for i in data]
    return Dataset(examples, fields)

def loadDataset():
    """
    Loading and Spliting Dataset
    """
    train, test = [], []
    
    print("==> Loading Dataset")
    
    with open("./data/Final_Dataset.json") as json_file:
        dataset = json.load(json_file)
    
    print("==> Preprocessing Data")
    
    dataset = extract_QnA_Ans_Sent(dataset)
    
    print("Size of Dataset : {}".format(len(dataset)))
    #test = extract_QnA_Ans_Sent(test)
    
    print("==> Creating Training Set and Test Set")
    
    train, test = train_test_split(dataset, test_size=0.2, shuffle = True)
    #train, val = train_test_split(train, test_size=0.2)
    
    fields_tuple = [('Q',Q),('Q_POS',Q_Pos_Tag),('A',A),('A_POS',A_Pos_Tag),('Ans_Sen',Ans_Sen), ('Ans_Sen_POS',A_Pos_Tag)]
    
    train_data = make_torchtext(train,fields_tuple)
    test_data = make_torchtext(test,fields_tuple)
    
    print("Size of Training set : {}".format(len(train_data)))
    print("Size of Test set : {}".format(len(test_data)))
    #val_data = make_torchtext(val,fields_tuple)
    
    print("==> Building Vocabulary using Fasttext")
    
    Q.build_vocab(train_data, specials = ['<sep>'], vectors = 'fasttext.simple.300d')
    A.build_vocab(train_data, vectors = 'fasttext.simple.300d')
    Ans_Sen.build_vocab(train_data, vectors = 'fasttext.simple.300d')
    Q_Pos_Tag.build_vocab(train_data)
    A_Pos_Tag.build_vocab(train_data)
    
    QnA_vocab = merge_vocabs([Q.vocab,A.vocab,Ans_Sen.vocab])
    Pos_vocab = merge_vocabs([Q_Pos_Tag.vocab,A_Pos_Tag.vocab])
    
    return train_data, test_data, QnA_vocab, Ans_Sen.vocab, Pos_vocab, fields_tuple

train_data, test_data, QnA_vocab, Ans_Sen_vocab, Pos_Tag_vocab, fields_tuple = loadDataset()
print(Pos_Tag_vocab.vectors)

# -*- coding: utf-8 -*-

