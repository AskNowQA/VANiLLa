import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 layers,
                 dropout,
                 pad_id,
                 device,
                 weights):
        super().__init__()
        
        self.device = device
        self.pad_id = pad_id
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_id)
        self.embedding.weight.data.copy_(weights)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=layers, dropout= dropout, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim, bias= True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        """
        src: (src_len, batch)
        src_len : (batch)
        """
        #src= self.dropout(src)
        #src = nn.utils.rnn.pad_sequence(src, padding_value= self.pad_id)
        #print(src.shape)
        
        embedded = self.dropout(self.embedding(src)) #(src_len, batch, embedding)
        #print(embedded.shape)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        #print(packed_embedded)
        #outputs, hidden = self.rnn(embedded)
        
        packed_outputs, hidden = self.rnn(packed_embedded)
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value = self.pad_id)
        outputs = self.dropout(outputs) #(src_len, batch, hidden_size * num_directions)
        #print(outputs.shape)
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) #(batch, enc_hid_dim)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, device):
        super().__init__()
        
        self.device = device
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        #self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
         
    def forward(self, hidden, encoder_outputs, mask):
        """
        encoder_outputs: (src_len, batch, hidden_size * num_directions)
        hidden :(batch, enc_hid_dim)
        mask : (batch, src_len)
        """
        
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
       
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) #(batch, src_len, dec_hid_dim)
        
        encoder_outputs = encoder_outputs.permute(1,0,2) #(batch, src_len, enc_hid_dim * 2)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) #(batch, src_len, dec_hid_dim)
        energy = energy.permute(0,2,1) #(batch, dec_hid_dim, src_len)
       
        v = self.v.repeat(batch_size, 1).unsqueeze(1) #(batch, 1, dec_hid_dim)
        
        attention = torch.bmm(v,energy).squeeze(1)
        #attention = self.v(energy).squeeze(2)
        #print(attention.shape)
        attention = attention.masked_fill(mask == 0, float('-inf'))
        
        return F.softmax(attention, dim=1) #(batch, src_len)
    
    
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 emb_dim, 
                 enc_hid_dim, 
                 dec_hid_dim, 
                 dropout,
                 pad_id,
                 device,
                 weights,
                 attention):
        super().__init__()
        
        self.device = device
        
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx = pad_id)
        self.embedding.weight.data.copy_(weights)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        """
        input : (trg_len-1 , batch)
        encoder_outputs: (src_len, batch, hidden_size * num_directions)
        hidden :(batch, enc_hid_dim)
        mask : (batch, src_len)
        """
        input = input.unsqueeze(1) #(trg_len-1 , 1, batch)
            
        embedded = self.dropout(self.embedding(input)).permute(1,0,2) #(1, batch, embedding)
        
        a = self.attention(hidden, encoder_outputs, mask)
        a = self.dropout(a)
        a = a.unsqueeze(1) #(batch, 1, src_len)
        #print(a.shape)
       
        encoder_outputs = encoder_outputs.permute(1, 0, 2) #(batch, src_len, hidden_size * num_directions)
        #print(encoder_outputs.shape)
        
        weighted = torch.bmm(a, encoder_outputs) 
        weighted = weighted.permute(1, 0, 2) #(1, batch, hidden_size * num_directions)
        
        #print(weighted.shape)
        #print(embedded.shape)
        
        rnn_input = torch.cat((embedded, weighted), dim = 2) #(1, batch, 2 * enc_hid_dim + embedding)
        
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0)) #(1, batch, dec_hid_dim),(1, batch, dec_hid_dim)
        
        #seq len, n layers and n directions will always be 1 in this decoder
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0) #(batch, embedding)
        output = output.squeeze(0) #(batch, dec_hid_dim)
        weighted = weighted.squeeze(0) #(batch, hidden_size * num_directions)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1)) #(batch, vocab)
            
        return prediction, hidden.squeeze(0), a.squeeze(1) #(batch, vocab), (batch, dec_hid_dim), (batch, src_len)
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_id, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.device = device
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        """
        src : (src_len, batch)
        
        trg : (trg_len, batch)
        """
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        #print(src.shape) (B,L)
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) #(trg_len, batch, vocab)
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device) #(trg_len, batch, src_len)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len) #(src_len, batch, 2 * hidden_size), (batch, enc_hid_dim)
        #print(encoder_outputs.shape)
        #print(hidden.shape)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        mask = (src != self.pad_id).permute(1,0) #(batch, src_len)
        #print(mask.shape)
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask) #(batch, vocab), (batch, dec_hid_dim), (batch, src_len)
            #print(output.shape)
            #print(hidden.shape)
            #print(attention.shape)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            attentions[t] = attention
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs #(trg_len, batch, vocab)
    


