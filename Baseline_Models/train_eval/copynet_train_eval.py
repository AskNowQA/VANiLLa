import torch
import torch.nn as nn
import spacy


def copynet_train(model, iterator, optimizer, criterion, clip):
    
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
        Q = nn.utils.rnn.pad_sequence(Q, padding_value= model.pad_id, batch_first = True)
        A = nn.utils.rnn.pad_sequence(A, padding_value= model.pad_id, batch_first = True)
        Ans_Sen = nn.utils.rnn.pad_sequence(Ans_Sen, padding_value= model.pad_id, batch_first = True) #(batch,trg_len)
        QnA = torch.cat((Q,A), 1) #(batch)
        
        #print(QnA.shape)
        #print(Ans_Sen.shape)
        
        #print(QnA.shape)
        #print(QnA_len)
        #print(Ans_Sen.shape)
        #Ans_Sen = Ans_Sen.permute(1,0)
        #QnA = QnA.permute(1,0)
        
        optimizer.zero_grad()
        
        output = model(QnA, Ans_Sen) #(batch, trg_len, vocab)
        #print(output.shape)
        """l = [Ans_Sen.shape[1]-1] * BATCH_SIZE
        print(l)
        target = nn.utils.rnn.pack_padded_sequence(Ans_Sen,l, batch_first=True)[0]
        pad_out = nn.utils.rnn.pack_padded_sequence(output,l, batch_first=True)[0]"""
        output_dim = output.shape[-1] #vocab
        #print(output_dim)
        
        output = output.permute(1,0,2)[1:].contiguous().view(-1, output_dim) #[(trg len - 1) * batch size, vocab]
        #print(output.shape)
        #print(Ans_Sen.shape)

        trg = Ans_Sen.permute(1,0)[1:].contiguous().view(-1) #[(trg len - 1) * batch size]
        #print(trg.shape)

        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def copynet_eval(model, iterator, criterion):
    
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
            Q = nn.utils.rnn.pad_sequence(Q, padding_value= model.pad_id, batch_first = True)
            A = nn.utils.rnn.pad_sequence(A, padding_value= model.pad_id, batch_first = True)
            Ans_Sen = nn.utils.rnn.pad_sequence(Ans_Sen, padding_value= model.pad_id, batch_first = True) #(batch,trg_len)
            QnA = torch.cat((Q,A), 1) #(batch, src_len)
            
            #print(QnA.shape)
            #print(Ans_Sen.shape)
            #print(QnA_len)
            #print(Ans_Sen.shape)
            #Ans_Sen = Ans_Sen.permute(1,0)
            #QnA = QnA.permute(1,0)
        
            output = model(QnA, Ans_Sen, 0) #turn off teacher forcing, (trg_len, batch, vocab)
            #print(output)
            output_dim = output.shape[-1]
        
            output = output.permute(1,0,2)[1:].contiguous().view(-1, output_dim) #[(trg len - 1) * batch size, vocab]
            trg = Ans_Sen.permute(1,0)[1:].contiguous().view(-1) #[(trg len - 1) * batch size]

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def copynet_predict(question, answer, q_field, a_field, as_field, QnA_vocab, model, device, max_len = 50):

    model.eval() 
    
    #(batch = 1)
    
    hid_dim = model.encoder.hid_dim
        
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
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device) #(1, src_len)
    #print(src_tensor.shape)
    #src_len = torch.LongTensor([len(src_indexes)]).to(device)
    trg_indexes = [as_field.vocab.stoi[as_field.init_token]]
    #trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    #print(trg_tensor.shape)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor) #(1, scr_len, hid*2), (direction, 1, hid)
        
    decoder_in, s, w = model.decoder_initial(src_tensor.size(0))
    #decoder_in = trg_indexes[:,0]

    #mask = (src_tensor != model.pad_id).type(torch.FloatTensor)*(-1000) #.permute(1,0)

    for i in range(max_len):

        #trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device) #(1)
        #print(trg_tensor.shape)
                
        with torch.no_grad():
            if i==0:
                output, s, w = model.decoder(decoder_in, encoder_outputs, src_tensor, s, w, i)
                #output = temp_output
            else:
                output, s, w = model.decoder(decoder_in, encoder_outputs, src_tensor, s, w, i)
                #output = output.permute(1,0,2)
                #output = torch.cat([output,temp_output],dim=1)
                #print(output.shape)
        
        
        #attentions[i] = attention
        #output = output.permute(1,0,2)
                #print(output.shape)
        output_dim = output.shape[-1]
        
        out = output.permute(1,0,2).contiguous().view(-1, output_dim)
        #print(output)
        
        pred_token = out.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == as_field.vocab.stoi[as_field.eos_token]:
            break
    
    trg_tokens = [as_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:] 


