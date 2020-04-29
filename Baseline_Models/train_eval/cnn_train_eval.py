import torch
import torch.nn as nn
import spacy


def cnn_train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        Q, Q_len = batch.Q
        A, A_len = batch.A
        Ans_Sen = batch.Ans_Sen
        
        Q = nn.utils.rnn.pad_sequence(Q, padding_value= model.pad_id, batch_first = True)
        A = nn.utils.rnn.pad_sequence(A, padding_value= model.pad_id, batch_first = True)
        Ans_Sen = nn.utils.rnn.pad_sequence(Ans_Sen, padding_value= model.pad_id, batch_first = True) #(batch,trg_len)
        QnA = torch.cat((Q,A), 1) #(batch,src_len)
        
        optimizer.zero_grad()
        
        output, _ = model(QnA, Ans_Sen[:,:-1])
        
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim)
        trg = Ans_Sen[:,1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def cnn_eval(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            Q, Q_len = batch.Q
            A, A_len = batch.A
            Ans_Sen = batch.Ans_Sen
            
            Q = nn.utils.rnn.pad_sequence(Q, padding_value= model.pad_id, batch_first = True)
            A = nn.utils.rnn.pad_sequence(A, padding_value= model.pad_id, batch_first = True)
            Ans_Sen = nn.utils.rnn.pad_sequence(Ans_Sen, padding_value= model.pad_id, batch_first = True) #(trg_len,batch)
            QnA = torch.cat((Q,A), 1) #(src_len, batch)

            output, _ = model(QnA, Ans_Sen[:,:-1])
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = Ans_Sen[:,1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def cnn_predict(question, answer, q_field, a_field, as_field, QnA_vocab, model, device, max_len = 50):

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

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(src_tensor)

    trg_indexes = [as_field.vocab.stoi[as_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == as_field.vocab.stoi[as_field.eos_token]:
            break
    
    trg_tokens = [as_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention
