# -*- coding: utf-8 -*-

import torch
import random
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_dim,
                 emb_dim,
                 hid_dim,
                 pad_id,
                 device,
                 weights):
        super().__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.embed = nn.Embedding(input_dim, emb_dim, padding_idx=pad_id)
        self.embed.weight.data.copy_(weights)
        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first = True, bidirectional = True)
        
    def forward(self, src):
        """
        src: [batch x src_len]
        
        """
        embedded = self.embed(src)
        out, h = self.rnn(embedded) # (batch, scr_len, hid*2), (direction, batch, hid)
        #print (out.shape)
        return out, h

class Decoder(nn.Module):
    
    def to_cuda(self, tensor): 
        if torch.cuda.is_available():
            #torch.cuda.device(2)
            return tensor.cuda()
        else:
            return tensor

    def __init__(self, output_dim, 
              emb_dim, 
              hid_dim,
              device):
    		super().__init__()
    		self.output_dim = output_dim
    		self.hid_dim = hid_dim
    		self.embed = nn.Embedding(output_dim, emb_dim)
    		self.gru = nn.GRU(emb_dim+(hid_dim*2),hid_dim, batch_first=True)
    		self.device = device
    		# weights
    		self.Ws = nn.Linear(hid_dim*2, hid_dim)
    		self.Wo = nn.Linear(hid_dim, output_dim) # generate mode
    		self.Wc = nn.Linear(hid_dim*2, hid_dim) # copy mode
    		self.nonlinear = nn.Tanh()
    
    def forward(self, input, encoded, src, prev_state, weighted, order):
         """
         input : (batch)
         encoded : ((batch, src_len, hid*2))
         src : (batch, src_len)
         prev_state : (batch, hid)
         weighted : (batch, hid*2)
         
         """
         #print(weighted.shape)
         #print(input.shape)
        
		# hyperparameters
         batch = encoded.size(0) # batch size
         src_len = encoded.size(1) # input sequence length
         #print(src_len)
         output_dim = self.output_dim
         hid_dim = self.hid_dim
         
         if order==0:
             prev_state = self.Ws(encoded[:,-1])
             weighted = torch.Tensor(batch,1,hid_dim*2).zero_()
             weighted = weighted.to(self.device)
             weighted = Variable(weighted)
             

         else:
             weighted = weighted.unsqueeze(1)
             weighted = weighted.to(self.device)
             weighted = Variable(weighted)
# (batch, 1, hidden*2)
         #weighted = Variable(weighted) 
        
         prev_state = prev_state.unsqueeze(0) # (1, batch, hid_dim)
         #print(input.shape)
         #print(encoded.shape)
         #print(src.shape)
         #print(prev_state.shape)
         #print(weighted.shape)
         #print(order)

         # 1. update states
         embedding = self.embed(input).unsqueeze(1) # (batch, 1, emb)
         #print(embedding.shape)
         gru_input = torch.cat([embedding, weighted],2) # (batch x 1 x (hid_dim*2+emb_dim))
         #print(gru_input.shape)
         _, hidden = self.gru(gru_input, prev_state) 
         hidden = hidden.squeeze(0) # (batch x hid_dim)
         #print(hidden.shape)
         # 2. predict next word y_t
         # 2-1) get scores score_g for generation- mode
         score_g = self.Wo(hidden) # [batch x output_dim]
         #print(score_g.shape)
		# 2-2) get scores score_c for copy mode, remove possibility of giving attention to padded values
         #print(encoded.contiguous().view(-1,hid_dim*2).shape)
         score_c = F.tanh(self.Wc(encoded.contiguous().view(-1,hid_dim*2))) # [b*seq x hidden_size]
         #print(score_c.shape)
         score_c = score_c.view(batch,-1,hid_dim) # [b x seq x hidden_size]
         score_c = torch.bmm(score_c, hidden.unsqueeze(2)).squeeze() # [b x seq]
         
         encoded_mask_np = np.array(src.cpu()==0, dtype=float)*(-1000)
         #print(encoded_mask_np)
         encoded_mask = torch.Tensor(encoded_mask_np)
         encoded_mask = encoded_mask.to(self.device) # [b x seq]
         encoded_mask = Variable(encoded_mask)
         score_c = score_c + encoded_mask # padded parts will get close to 0 when applying softmax
         score_c = torch.tanh(score_c) # purely optional....
         
         # 2-3) get softmax-ed probabilities
         score = torch.cat([score_g,score_c],1) # [b x (vocab+seq)]
         probs = F.softmax(score)
         prob_g = probs[:,:output_dim] # [b x vocab]
         prob_c = probs[:,output_dim:] # [b x seq]
         # remove scores which are obsolete
         #print(score.shape)
         #print(probs.shape)
         #print(prob_g.shape)
         #print(prob_c.shape)


         # 2-4) add prob_c to prob_g
         prob_c_to_g = torch.Tensor(batch,output_dim).zero_().to(self.device)
         prob_c_to_g = Variable(prob_c_to_g)
         for b_idx in range(batch): # for each sequence in batch
             for s_idx in range(src_len):
                 prob_c_to_g[b_idx,src[b_idx,s_idx]]=prob_c_to_g[b_idx,src[b_idx,s_idx]]+prob_c[b_idx,s_idx]            
         predicted = prob_g + prob_c_to_g
         predicted = predicted.unsqueeze(1)
         idx_from_input = []
         #print(predicted.shape)
         for i,j in enumerate(src):
             idx_from_input.append([int(k==input[i].item()) for k in j])
         idx_from_input = torch.Tensor(np.array(idx_from_input, dtype=float))
         idx_from_input = idx_from_input.to(self.device)
         #print(idx_from_input.shape)
         # idx_from_input : np.array of [b x seq]
         idx_from_input = Variable(idx_from_input)
         for i in range(batch):
             if idx_from_input[i].sum().item()>1:
                 idx_from_input[i] = idx_from_input[i]/idx_from_input[i].sum().item()
         # 3-2) multiply with prob_c to get final weighted representation
         attn = prob_c * idx_from_input
         attn = attn.unsqueeze(1)
         #print(attn.shape)
         #print(encoded.shape)
         weighted = torch.bmm(attn, encoded) # weighted: [b x 1 x hidden*2]weighted = weighted.squeeze(1)
         weighted = weighted.squeeze(1)
         
         return predicted, hidden, weighted
     
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_id, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id
        self.device = device
        
        self.Ws = nn.Linear(self.encoder.hid_dim*2, self.encoder.hid_dim)
    
        
    def decoder_initial(self, batch_size):
        decoder_in = torch.LongTensor(np.ones(batch_size,dtype=int))*2
        s = None
        w = None
        decoder_in = decoder_in.to(self.device)
        decoder_in = Variable(decoder_in)
        return decoder_in, s, w
    
    def numpy_to_var(x,is_int=True):
        if is_int:
            x = torch.LongTensor(x)
        else:
            x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        """
        src : (batch, src_len)
        trg: (batch, trg_len)
        """
        
        trg_len = trg.shape[1]
        #print(trg_len)
       
        #outputs = torch.zeros(trg_len, batch, trg_vocab_size).to(self.device)
        
        encoded, _ = self.encoder(src) #(batch, src_len, hid*2), (direction, batch, hid)
        #print(output.shape)
        #print(hidden.shape)
        
        decoder_in, s, w = self.decoder_initial(src.size(0))
        #decoder_in = trg[:,0]
        
        #prev_state = self.Ws(output[:,-1]) #(batch, hid)
        
        #print(prev_state.shape) 
			
        #weighted = torch.Tensor(batch,hid_dim*2).zero_() #(batch, hid*2)
        #print(weighted.shape) 
        #mask = (src != self.pad_id).type(torch.FloatTensor)*(-1000)
        #order = 0 
        
        #input = trg[:,0] #(batch)
        #print(input.shape)
        out_list= []
        for t in range(0, trg_len):
            
            if t==0:
                out, s, w = self.decoder(decoder_in, encoded,
                                src, s, w, t)
            else:
                tmp_out, s, w = self.decoder(decoder_in, encoded,
                                src, s,
                                w, t)
                out = torch.cat([out,tmp_out],dim=1)
            
            #predicted, prev_state, weighted = self.decoder(input, output, src, prev_state, weighted, mask)
            #print(predicted)
            #outputs[t] = predicted
            teacher_force = random.random() < teacher_forcing_ratio
            
            if teacher_force:
                decoder_in = out[:,-1].max(1)[1].squeeze()
            else:
                decoder_in = trg[:,t]
            out_list.append(out[:,-1].max(1)[1].squeeze().cpu().data.numpy())
        #print(out.shape)
        #print(t)
        #out = out.permute(1,0,2)

        return out #(batch, trg_len, vocab)
                                        	
