import torch
import torch.nn as nn
import pdb
from utils.functions import TimeDistributed

class tr_beam_search_step():
# tr is batch decoding
    # local attention
    # pl=t -- monotonic alignment
    # pl=? Predeictive alignment pl in [0,T]
    D:window_size
    decode_step:l, L
    encode_step:t, T 
    def forward_step(self, previous_y, last_context, 
        last_sl_hidden, encoder_outputs, src_lens, step, D):
        rnn_input = 
        pl = src_lens.mul(nn.Sigmoid(nn.tanh)) # predict the position




def inf_beam_search_step():
# inf is inference decoding so is sample decoding
# force the decoded sequence is longer than the reference 


# lens_src lens_trg
# manlen_enc maxlen_dec
# dim_enc dim_dec
# post_keys post_query :post processed
# bsize: batch_size
# query: state of dec
# keys: state of enc
class Attention(nn.Module):
    '''
    This is location-aware content based mechanism
    '''
    def __init__(self, num_kernels, temperature, 
        dim_dec, 
        num_kernels, smooth=False,
        flag_unified=False, dim_unified=100,
        flag_window=True, window_size=10,
        mode, device='cpu')
        super(Attention).__init__()
        self.prev_attn = None
        self.smooth = smooth
        self.device = device
        self.temperature = temperature
        self.w = window_size
        dim = dim_enc
        if flag_unified:
            dim = dim_unified
            self.linear_enc = nn.Linear(dim_enc, dim, bias=False)
            self.linear_dec = nn.Linear(dim_dec, dim, bias=False)
            
        self.filter = nn.Conv1d(1, num_kernels, 1, bias=False) #in_channel, out_channel, kernel_size(tuple or int), stride 
        self.linear_u = nn.Linear(num_kernels, dim)
        self.linear_w = nn.Linear(dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def _attend(self, energy, mask, window):
        attn = energy / self.temperature
        if flag_window:
            attn = attn.masked_fill(window, -2e18)
        attn = attn.masked_fill(mask, -2e18)
        if self.smooth:
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
        values = keys
        # compute mask
        mask = []
        bsize, maxlen_enc, _ = keys.shape
        for b in range(bsize):
            mask.append([0]*lens_src[b].item() + [1]*(maxlen_enc - lens_src[b].item()))
        mask = torch.ByteTensor(mask).to(self.device)
        # compute window
        window=None
        if flag_window:
            window = []
            pl = torch.median(self.prev_attn, -1)[1] #(B,1)
            for b in range(bsize): 
                if pl[b].item()-self.w <= 0:
                    window.append([0]*(self.w+1+pl[b].item())+[1]*(maxlen_enc-pl[b].item()-self.w-1))
                elif (maxlen_enc-(pl[b].item()-self.w)-2*self.w-1) <= 0:
                    window.append([1]*(pl[b].item()-self.w)+[0]*(maxlen_enc+self.w-pl[b].item()))
                else:                 
                    window.append([1]*(pl[b].item()-self.w)+[0]*(2*self.w+1)+[1]*(maxlen_enc-pl[b].item()-self.w-1))
            window = torch.ByteTensor(window).to(self.device)
        if flag_unified:
        	keys = TimeDistributed(self.linear_enc, keys)
        	query = self.linear_dec(query)
        query = query.repeat(1, maxlen_enc, 1) #(B,1,dim)-->(B,T,dim)
        # compute energy
        if mode=='dot':
        	energy = torch.bmm(query[:,0:1,:], keys.transpose(1,2)) # (B,1,dim)*(B,dim,T)-->(B,1,T)
        elif mode=='concat':
        	energy = TimeDistributed(self.linear_w, torch.tanh(query+keys))
        elif mode=='conv':
            conv_fea = TimeDistributed(self.linear_u, self.filter(self.prev_attn.unsqueeze(1).transpose(1,2))) #(B,1,T)-->(B,T,K)-->(B,T,dim)            
            energy = TimeDistributed(self.linear_w, torch.tanh(query+keys+conv_fea))
        
        attn = self._attend(energy, mask, window)
        self.prev_attn = attn
        context = torch.bmm(attn, values) #(B,1,T)*(B,T,dim)-->(B,1,dim)
        return context, attn



class Decoder(nn.Module):
    def __init__(self, dim_tokens,  
        dim_enc, encoder_directions, 
        dim_dec, decoder_layer, 
        maxlen_dec, flag_transducer,
        flag_unified, dim_unified,
        dim_embed,
        rnn_unit, attention_mode='concat',
        decode_mode=1, dropout_rate=0.0, 
        device="cpu", **kwargs):
        super(Decoder, self).__init__()
        self.device = device
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        self.maxlen_dec = maxlen_dec
        self.flag_transducer = flag_transducer
        self.decode_mode = decode_mode
        # self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.dim_tokens = dim_tokens
        self.dim_embed = dim_embed       
        self.embedding = nn.Embedding(dim_tokens, dim_embed)
        self.rnn_layer = self.rnn_unit(dim_embed+dim_enc*encoder_directions,
            dim_dec, num_layers=decoder_layer, dropout=dropout_rate,
            batch_first=True)
            # 
        self.attention = Attention(dim_enc*encoder_directions,
            dim_dec, flag_unified, dim_unified, mode=attention_mode, 
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
        rnn_input = torch.cat([prev_pred.to(self.device), prev_context], dim=-1) #(B,1,dim)
        rnn_output, sl = self.rnn_layer(rnn_input, prev_sl)
        attn, context = self.attention(rnn_output, encoder_outputs, lens_src)
        if self.flag_transducer:
            concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1) # [B, 50+100]
            prob = self.softmax(self.phone_distribution(concat_feature)) #[B, output_size]
        else:
            prob = self.softmax(self.phone_distribution(rnn_output.squeeze(dim=1)))
        return prob, sl, context, attn
# 
    def forward(self, encoder_outputs, lens_src, ground_truth=None, teacher_force_rate=0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
# 
        batch_size = encoder_outputs.size(0)
#         
        pred = CreateEmbedVariable(
            torch.LongTensor(np.zeros((batch_size, 1))).to(self.device), 
            self.dim_tokens, self.dim_embed)
        # pred0 context0 
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
            prob, sl, context, attn = self.forward_step(prev_y, prev_context, prev_sl, encoder_outputs, lens_src)
            batch_size = prob.shape[0]
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
            pred = CreateEmbedVariable(raw_pred.to(self.device), 
                self.token_dim, self.embed_dim)  
            # 
        return seq_prob, seq_attn



def forced_aliIters(loader, preprocessor, encoder, decoder):
    '''
    bach_size is 1
    '''
    device = encoder.device
    encoder.eval()
    decoder.eval()
    output = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, x_len, y_len, utt = collate(*batch)
            x = x.to(device); x_len = x_len.to(device)
            y = y.to(device); y_len = y_len.to(device) #y.shape[1] = y_len+2
            true_y = preprocessor.decode(y[0].contiguous()) #[B, len(y)]
            # 
            batch_size = x.size(0)
            maxlen_dec = min(y.size(1), decoder.maxlen_dec)
            # 
            encoder_output = encoder(x, x_len)
            raw_prob, attn= decoder(encoder_output, x_len, ground_truth=y,
                teacher_force_rate=1.2)
            # 
            raw_prob = (torch.cat([each_y.unsqueeze(1) for each_y in raw_prob], 1)[:,:y_len,:]).contiguous()  #[B, max_token_len-1, num_classes] since the start_token is skip
            pred = preprocessor.forced_decode(torch.max(raw_prob, 2)[1][0])
            att = (torch.cat([each_att[0].unsqueeze(1) for each_att in attn], 1)[:,:y_len,:]).contiguous()
            output[utt] = {}
            output[utt]['att'] = att
            output[utt]['pred'] = pred
            output[utt]['prob'] = prob
            output[utt]['truth'] = true_y
            # 
    return output



