from train_eval.cnn_train_eval import *
from data.dataset import *
from models.cnn_model import *
from utils.bleu_evaluator import *
from utils.constants import *

from torchtext.data import BucketIterator, Dataset
from sklearn.model_selection import KFold

import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def iter_folds(data, fields):
    train_exs_arr = np.array(data)
    splits = list(kf.split(train_exs_arr))
    #print(splits)
    for train_idx, val_idx in splits:
        
        train_set = Dataset(train_exs_arr[train_idx], fields)
        #print(len(train_exs_arr[train_idx]))
        #print(train_set[0].__dict__.keys())
        val_set = Dataset(train_exs_arr[val_idx], fields)
        #print(val_set[0:].__dict__.keys())
        
        yield (
            train_set,
            val_set,
            )

def get_iterator(dataset, batch_size, device, train=True, shuffle=True, repeat=False):
    dataset_iter = BucketIterator(
        dataset, batch_size=batch_size, 
        train=train, shuffle=shuffle, repeat=repeat,
        sort_key=lambda x: len(x.Q)+len(x.A), 
        sort_within_batch = True, 
        device = device)
    return dataset_iter

train_data, test_data, QnA_vocab = loadDataset()

fields = [('Q',Q),('A',A),('Ans_Sen',Ans_Sen)]

lr = LR
INPUT_DIM = len(QnA_vocab)
OUTPUT_DIM = len(QnA_vocab)
TRG_PAD_IDX = QnA_vocab.stoi[Ans_Sen.pad_token]

kf = KFold(n_splits=N_FOLDS, shuffle = True, random_state=SEED)

print("==> Building Encoder")

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, CNN_ENC_DROPOUT, device, QnA_vocab.vectors)

print("==> Building Decoder")

dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, CNN_DEC_DROPOUT, TRG_PAD_IDX, device, QnA_vocab.vectors)

print("==> Building Seq2Seq Model")

model = Seq2Seq(enc, dec, TRG_PAD_IDX, device).to(device)

#train_iter, val_iter, test_iter = BucketIterator.splits((train_data, val_data, test_data), batch_size = BATCH_SIZE, sort_key=lambda x: len(x.Q), sort_within_batch = True, device = device)

optimizer = optim.Adam(model.parameters())#,lr=lr,weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

print("The model has {} trainable parameters".format(count_parameters(model)))

best_valid_loss = float('inf')

print("==> Start Training...")

for epoch in range(CNN_EPOCHS):
    
    print(" Start of Epoch: {:02}".format(epoch+1))
    start_time = time.time()
    
    train_val_generator = iter_folds(train_data.examples, fields)
    
    for fold, (train_dataset, val_dataset) in enumerate(train_val_generator):
        
        print("\tFold: {:02}".format(fold+1))
        
        train_iter = get_iterator(train_dataset, BATCH_SIZE, device)
    
        train_loss = cnn_train(model, train_iter, optimizer, criterion, CLIP)
        
        print("\tTrain Loss: {:.3f} | Train PPL: {:7.3f}".format(train_loss,math.exp(train_loss)))
        
        val_iter = get_iterator(val_dataset, BATCH_SIZE, device) 
        
        valid_loss = cnn_eval(model, val_iter, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'checkpoints/cnn-model-extended.pt')
    
        print("\t\tVal. Loss: {:.3f} |  Val. PPL: {:7.3f}".format(valid_loss,math.exp(valid_loss)))
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    
    print("End of Epoch: {:02} | Time: {}m {}s".format(epoch+1,epoch_mins,epoch_secs))

print("==> Start Testing...")

test_iter = get_iterator(test_data, BATCH_SIZE, device)    
    
model.load_state_dict(torch.load('checkpoints/cnn-model-extended.pt'))

test_loss = cnn_eval(model, test_iter, criterion)

print("| Test Loss: {:.3f} | Test PPL: {:7.3f} |".format(test_loss,math.exp(test_loss)))

print("==> Start Evaluation...")

bleu_score, precision = calculate_bleu("CNN", test_data, Q, A, Ans_Sen, QnA_vocab, model, device)

print('BLEU score = {:.2f}'.format(bleu_score*100))
print('Precision = {:.2f}'.format(precision*100))


