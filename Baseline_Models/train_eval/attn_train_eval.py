import torch
import torch.nn as nn
import spacy


def attn_train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    #print(iterator)
    
    for i, batch in enumerate(iterator):
        
        #print(batch)
        
        Q, Q_len = batch.Q
        A, A_len = batch.A
        Ans_Sen = batch.Ans_Sen
        #print(Q_len)
        #print(A_len)
        #print(Q.shape) 
        #print(A.shape)
        #print(Ans_Sen.shape)
        Q = nn.utils.rnn.pad_sequence(Q, padding_value= model.pad_id, batch_first = False)
        A = nn.utils.rnn.pad_sequence(A, padding_value= model.pad_id, batch_first = False)
        Ans_Sen = nn.utils.rnn.pad_sequence(Ans_Sen, padding_value= model.pad_id, batch_first = False) #(trg_len,batch)
        QnA = torch.cat((Q,A), 0) #(src_len, batch)
        QnA_len = torch.empty(QnA.shape[1]) 
        QnA_len.fill_(QnA.shape[0])#(batch)
        #print(QnA.shape)
        #print(QnA_len)
        #print(Ans_Sen.shape)
        #Ans_Sen = Ans_Sen.permute(1,0)
        #QnA = QnA.permute(1,0)
        
        optimizer.zero_grad()
        
        output = model(QnA, QnA_len, Ans_Sen) #(trg_len, batch, vocab)
        
        output_dim = output.shape[-1] #vocab
        
        output = output[1:].view(-1, output_dim) #[(trg len - 1) * batch size, vocab]

        trg = Ans_Sen[1:].contiguous().view(-1) #[(trg len - 1) * batch size]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def attn_eval(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            Q, Q_len = batch.Q
            A, A_len = batch.A
            Ans_Sen = batch.Ans_Sen
            #print(Q_len)
            #print(A_len)
            #print(Q.shape) 
            #print(A.shape)
            #print(Ans_Sen.shape)
            Q = nn.utils.rnn.pad_sequence(Q, padding_value= model.pad_id, batch_first = False)
            A = nn.utils.rnn.pad_sequence(A, padding_value= model.pad_id, batch_first = False)
            Ans_Sen = nn.utils.rnn.pad_sequence(Ans_Sen, padding_value= model.pad_id, batch_first = False) #(trg_len,batch)
            QnA = torch.cat((Q,A), 0) #(src_len, batch)
            QnA_len = torch.empty(QnA.shape[1]) 
            QnA_len.fill_(QnA.shape[0])#(batch)
            #print(QnA.shape)
            #print(QnA_len)
            #print(Ans_Sen.shape)
            #Ans_Sen = Ans_Sen.permute(1,0)
            #QnA = QnA.permute(1,0)
        
            output = model(QnA, QnA_len, Ans_Sen,0) #turn off teacher forcing, (trg_len, batch, vocab)
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim) #[(trg len - 1) * batch size, vocab]
            trg = Ans_Sen[1:].contiguous().view(-1) #[(trg len - 1) * batch size]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def attn_predict(question, answer, q_field, a_field, as_field, QnA_vocab, model, device, max_len = 50):

    model.eval()
        
    if isinstance(question, str):
        nlp = spacy.load('en')
        q_tokens = [token.text.lower() for token in nlp(question)]
    else:
        q_tokens = [token.lower() for token in question]
        
    if isinstance(answer, str):
        nlp = spacy.load('en')
        a_tokens = [token.text.lower() for token in nlp(answer)]
    else:
        a_tokens = [token.lower() for token in answer]

    tokens = [q_field.init_token] + q_tokens + [q_field.eos_token] + a_tokens + [a_field.eos_token]  
    src_indexes = [QnA_vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
        
    trg_indexes = [as_field.vocab.stoi[as_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    mask = (src_tensor != model.pad_id).permute(1,0)

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention
            
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == as_field.vocab.stoi[as_field.eos_token]:
            break
    
    trg_tokens = [as_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

