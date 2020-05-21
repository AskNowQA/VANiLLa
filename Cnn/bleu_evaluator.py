import json

from nltk.translate.bleu_score import sentence_bleu, modified_precision, SmoothingFunction

from train_eval.cnn_train_eval import cnn_predict
from train_eval.attn_train_eval import attn_predict
from train_eval.copynet_train_eval import copynet_predict


def example_score(reference, hypothesis):
    """Calculate blue score for one example"""
    cc = SmoothingFunction()
    return sentence_bleu([reference], hypothesis,smoothing_function=cc.method4)


def calculate_bleu(name, dataset, q_field, a_field, as_field, QnA_vocab, model, device, max_len = 50):
    
    results = []
    precision_total = 0
    score = 0
    instances = len(dataset)
    
    for data in dataset:
        
        q = vars(data)['Q']
        a = vars(data)['A']
        a_sen = vars(data)['Ans_Sen']
        
        if name == "CNN":
            hypothesis, _ = cnn_predict(q, a, q_field, a_field, as_field, QnA_vocab, model, device, max_len)
        elif name == "ATTN":
            hypothesis, _ = attn_predict(q, a, q_field, a_field, as_field, QnA_vocab, model, device, max_len)
        elif name == "COPYNET" :
            hypothesis = copynet_predict(q, a, q_field, a_field, as_field, QnA_vocab, model, device, max_len)
        else :
            hypothesis = []
        #cut off <eos> token
        
        hypothesis = hypothesis[:-1]
        reference = a_sen
        #reference = [t.lower() for t in a_sen]

        blue_score = example_score(reference, hypothesis)
        precision = float(modified_precision([reference], hypothesis, n = 1))
        #print(precision)

        results.append({
            'question':q,
            'answer': a,
            'reference': reference,
            'hypothesis': hypothesis,
            'blue_score': blue_score,
            'precision': precision
        })
    
        score += blue_score
        precision_total +=precision
       
        
    if name == "CNN":
        with open('CNN_results-ext.txt', 'w') as file:
            file.write(json.dumps(results))
    
    elif name == "ATTN":
        with open('ATTN_results-ext.txt', 'w') as file:
            file.write(json.dumps(results))
            
    elif name == "COPYNET":
        with open('COPYNET_results.txt', 'w') as file:
            file.write(json.dumps(results))
            
    return score / instances, precision_total / instances
        
        
    #return bleu_score(pred_trgs, trgs, max_n = 2, weights = [0.5,0.5])
