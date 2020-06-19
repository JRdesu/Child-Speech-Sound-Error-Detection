import numpy as np
import torch
import torch.nn as nn
import pdb
from utils.functions import TimeDistributed
from torch.distributions.categorical import Categorical

class Attention(nn.Module):
    '''
    This is location-aware content based mechanism
    '''
    def __init__(self, temperature,
        dim_enc, dim_dec,
        dim_unified,
        num_kernels, 
        flag_smooth=False,
        flag_window=True, 
        window_size=10,
        mode='dot', device='cpu')
        super(Attention).__init__()
        self.prev_attn = None
        self.temperature = temperature
        self.flag_smooth = flag_smooth
        self.flag_window = flag_window
        self.w = window_size
        self.mode= mode
        self.device = device
        #unify the embedding spaces of enc and dec
        dim = dim_unified
        self.linear_enc = nn.Linear(dim_enc, dim, bias=False)
        self.linear_dec = nn.Linear(dim_dec, dim, bias=False)
            
        self.filter = nn.Conv1d(1, num_kernels, 1, bias=False) #in_channel, out_channel, kernel_size(tuple or int), stride 
        self.linear_u = nn.Linear(num_kernels, dim)
        self.linear_w = nn.Linear(dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    def _attend(self, energy, mask, window):
        attn = energy / self.temperature
        if self.flag_window:
            attn = attn.masked_fill(window, -2e18)
        attn = attn.masked_fill(mask, -2e18)
        if self.flag_smooth:
            attn = torch.sigmoid(attn)
            attn_sum = attn.sum(-1).repeat(1, attn.shape[-1]).unsqueeze(1).contiguous() #(B,1,T)
            attn = attn/attn_sum
        else:
            attn = self.softmax(attn)
        return attn #(B,1,T)
    def forward(self, query, keys, lens_src):
        '''
        query: decoder cell state (B,1,dim_dec)
        keys: encoder states (B,T,dim_enc)
        values: encoder states value; usually keys=values
        '''
        values = keys #dim_enc=2*dim_dec
        # compute mask
        mask = []
        batch_size, maxlen_enc, _ = keys.shape
        for b in range(batch_size):
            mask.append([0]*lens_src[b].item() + [1]*(maxlen_enc - lens_src[b].item()))
        mask = torch.ByteTensor(mask).to(self.device)
        # uniformly init prev_attn
        if self.prev_attn is None:
            self.prev_attn = torch.zeros(batch_size, 1, maxlen_enc).to(self.device)
            for b in range(batch_size):
                self.prev_attn[b,:,:lens_src[b]] = 1.0/lens_src[b]
        # compute window
        window=None
        if self.flag_window:
            window = []
            pl = torch.median(self.prev_attn, -1)[1] #(B,1)
            for b in range(batch_size): 
                if pl[b].item()-self.w <= 0:
                    window.append([0]*(self.w+1+pl[b].item())+[1]*(maxlen_enc-pl[b].item()-self.w-1))
                elif (maxlen_enc-(pl[b].item()-self.w)-2*self.w-1) <= 0:
                    window.append([1]*(pl[b].item()-self.w)+[0]*(maxlen_enc+self.w-pl[b].item()))
                else:                 
                    window.append([1]*(pl[b].item()-self.w)+[0]*(2*self.w+1)+[1]*(maxlen_enc-pl[b].item()-self.w-1))
            window = torch.ByteTensor(window).to(self.device) #(B,1,maxlen_enc)
        # unify the embedding spaces of keys and query
        keys = TimeDistributed(self.linear_enc, keys)
        query = self.linear_dec(query)
        query = query.repeat(1, maxlen_enc, 1) #(B,1,dim)-->(B,T,dim)
        # compute energy
        if self.mode=='dot':
            energy = torch.bmm(query[:,0:1,:], keys.transpose(1,2)) # (B,1,dim)*(B,dim,T)-->(B,1,T)
        elif self.mode=='concat':
            energy = TimeDistributed(self.linear_w, torch.tanh(query+keys))
        elif self.mode=='conv':
            conv_fea = TimeDistributed(self.linear_u, self.filter(self.prev_attn.unsqueeze(1).transpose(1,2))) #(B,1,T)-->(B,T,K)-->(B,T,dim)            
            energy = TimeDistributed(self.linear_w, torch.tanh(query+keys+conv_fea))
        
        attn = self._attend(energy, mask, window)
        self.prev_attn = attn
        context = torch.bmm(attn, values) #(B,1,T)*(B,T,dim)-->(B,1,dim)
        return context, attn 

class Decoder(nn.Module):
    def __init__(self, 
        dim_tokens,
        dim_embed,
        maxlen_dec, 
        rnn_unit,
        decode_mode,
        flag_transducer,
        dim_enc, encoder_directions, 
        dim_dec, decoder_layer,         
        temperature, num_kernels, dim_unified, flag_sooth, flag_window, window_size, attention_mode,
        dropout_rate=0.0, 
        device="cpu", **kwargs):
        super(Decoder, self).__init__()
        self.maxlen_dec = maxlen_dec
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        self.decode_mode = decode_mode
        self.flag_transducer = flag_window
        self.device = device
        # 
        self.embedding = nn.Embedding(dim_tokens, dim_embed)
        self.rnn_layer = self.rnn_unit(dim_embed+dim_enc*encoder_directions,
            dim_dec, num_layers=decoder_layer, dropout=dropout_rate,
            batch_first=True)
        self.attention = Attention(temperature, dim_enc*encoder_directions,
            dim_dec, num_kernels, dim_unified, flag_smooth, flag_window, window_size, attention_mode,  
            device=self.device)
        if self.flag_transducer:
            self.phone_distribution = nn.Linear(dim_dec+dim_enc*encoder_directions, 
            dim_tokens)
        else:
            self.phone_distribution = nn.Linear(dim_dec, dim_tokens)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.apply(_init_s2s_weights)
    # Stepwise operation of each sequence
    def forward_step(self, prev_pred, prev_context, prev_sl, encoder_outputs, lens_src):
        '''
        input_word: [batch_size, token_length] 
        '''
        rnn_input = torch.cat([prev_pred.to(self.device), prev_context], dim=-1) #(B,1,dim_embed+dim_dec*2)
        rnn_output, sl = self.rnn_layer(rnn_input, prev_sl)
        attn, context = self.attention(rnn_output, encoder_outputs, lens_src)
        if self.flag_transducer:
            concat_feature = torch.cat([rnn_output, context], dim=-1) # (B,1,dim_dec+dim_enc*2)
            prob = self.softmax(self.phone_distribution(concat_feature)) #(B,1,dim_tokens)
        else:
            prob = self.softmax(self.phone_distribution(rnn_output)) #(B,1,dim_tokens)
        return prob, sl, context, attn
# 
    def forward(self, encoder_outputs, lens_src, ground_truth=None, teacher_force_rate=0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
        batch_size = encoder_outputs.size(0)
        # init pred0 context0 sl0
        pred = self.embedding(torch.LongTensor(np.zeros((batch_size, 1))).to(self.device)) #(B,1)-->(B,1,dim_embed)
        context = encoder_outputs[:, 0:1, :] #[B, 1, encoder_dim]
        sl = None
# 
        seq_prob = [] # log probability
        seq_attn = []
        if (ground_truth is None) or (not teacher_force):
            max_step = self.maxlen_dec
        else:
            max_step = ground_truth.size()[1] #trg_len+2
        for step in range(1, max_step):
            prob, sl, context, attn = self.forward_step(pred, context, sl, encoder_outputs, lens_src)
            seq_prob.append(prob)
            seq_attn.append(attn)
            # Teacher force - use ground truth as next step's input
            if teacher_force:
                raw_pred = ground_truth[:, step:step+1] #[B, 1]
            else:
                # Case 1. Pick character with max probability
                if self.decode_mode == 1:
                    raw_pred = prob.topk(1)[1].squeeze(-1).contiguous().view(batch_size, 1) #[B, 1]                                
                # Case 2. Sample categotical label from raw prediction
                elif self.decode_mode == 2:
                    raw_pred = Categorical(prob).sample().view(batch_size,1)# [B, 1]
            pred = self.embedding(raw_pred.to(self.device))  
            # 
        return seq_prob, seq_attn
