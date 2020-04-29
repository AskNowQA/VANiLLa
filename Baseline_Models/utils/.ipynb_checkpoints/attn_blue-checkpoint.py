from torchtext.data.metrics import bleu_score
from train_eval.rnn_train_eval import *
from data.dataset import *
from models.rnn_model import *

def calculate_bleu(dataset, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for data in dataset:
        
        src = vars(data)['QnA']
        trg = vars(data)['Ans_Sen']
        
        print(trg)
        
        pred_trg, _ = cnn_predict(src, src_field, trg_field, model, device, max_len)
        
        print(pred_trg)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)
