import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.data import BucketIterator

import random
import math
import time
import spacy
import numpy as np

from models.rnn_model import *
from data.dataset import *

def rnn_train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        QnA, QnA_len = batch.QnA
        Ans_Sen = batch.Ans_Sen
        
        Ans_Sen = Ans_Sen.permute(1,0)
        QnA = QnA.permute(1,0)
        
        optimizer.zero_grad()
        
        output = model(QnA, Ans_Sen)
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = Ans_Sen[1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def rnn_eval(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            QnA, QnA_len = batch.QnA
            Ans_Sen = batch.Ans_Sen
            
            Ans_Sen = Ans_Sen.permute(1,0)
            QnA = QnA.permute(1,0)

            output = model(QnA, Ans_Sen,0) #turn off teacher forcing
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = Ans_Sen[1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def rnn_predict(sentence, src_field, trg_field, model, device, max_len = 50):

    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_hidden = model.encoder(src_tensor)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, encoder_hidden)

        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    print(trg_tokens)
    
    return trg_tokens[1:]

