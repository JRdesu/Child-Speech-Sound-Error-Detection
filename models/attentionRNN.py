import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb

'''
Reference:
https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq/blob/master/attentionRNN.py
'''
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.5):
#         super(EncoderRNN, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.dropout = dropout
#         # self.embedding = nn.Embedding(input_size,embed_size)
#         self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
# # 
#     def forward(self, input_seqs, input_lengths, hidden=None):
#         '''
#         :param input_seqs: 
#             Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
#         :param input:
#             list of sequence length
#         :param hidden:
#             initial state of GRU
#         :returns:
#             GRU outputs in shape (T,B,hidden_size(H))
#             last hidden stat of RNN(i.e. last output for GRU)
#         '''
#         packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths)
#         outputs, hidden = self.gru(packed, hidden)
#         outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
#         outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
#         return outputs, hidden

class DynamicEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0, device="cpu"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.gru = nn.GRU(input_size, hidden_size, n_layers, 
                          bidirectional=True)
        self.device = device
        self.apply(_init_siam_weights)
# 
    def forward(self, input_seqs, input_lens, hidden_0=None):
        """
        forward procedure. **No need for inputs to be sorted**
        :param input_seqs: Variable of [B,T,E]
        :param hidden_0:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(0)
        # embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-(input_lens.cpu()))
        # pdb.set_trace()
        unsort_idx = torch.LongTensor(np.argsort(sort_idx)).to(self.device)
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx).to(self.device)
        input_seqs = input_seqs[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lens)
        outputs, hidden = self.gru(packed, hidden_0) #outputs:[T,B,2H] hidden:[directions*layers, B, H]
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # [T, B, H]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous() # [T, B, H]
        hidden = hidden.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous() # [num_layers*num_directions, B, H]
        return outputs, hidden
        

class Attn(nn.Module):
    def __init__(self, method, hidden_size, device="cpu"):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        self.device = device
        self.apply(_init_siam_weights)
# 
    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score [B,T]
        # 
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            # mask = torch.ByteTensor(mask).unsqueeze(1).to(self.device) # [B,1,T]
            mask = torch.ByteTensor(mask).to(self.device) # [B,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        # 
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # normalize with softmax [B,1,T]
# 
    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0, device="cpu"):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size, self.device)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, 
            dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)       
        self.apply(_init_siam_weights)
# 
    def forward(self, word_input, last_hidden, encoder_outputs, src_len=None):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        # last_hidden = last_hidden[-1].view(1, last_hidden.size(1), last_hidden.size(2)).cont
        attn_weights = self.attn(last_hidden[-1], encoder_outputs, src_len)
        # attn_weights = self.attn(last_hidden, encoder_outputs, src_len)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # [B,1,T]*[B,T,H]-->(B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # pdb.set_trace()
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden[-1].unsqueeze(0))
        # output, hidden = self.gru(rnn_input, last_hidden.unsqueeze(0))
        # output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output),dim=1)
        # Return final output, hidden state
        return output, hidden, attn_weights.squeeze(1)

def _init_siam_weights(module):
    """
    Initialise the weights of the SIAM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
# 
    elif isinstance(module, nn.GRU):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 3
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0
# 
        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
# # 
#     elif isinstance(module, nn.RNN):
#         nn.init.xavier_uniform_(module.weight_ih_l0.data)
#         nn.init.orthogonal_(module.weight_hh_l0.data)
#         nn.init.constant_(module.bias_ih_l0.data, 0.0)
#         nn.init.constant_(module.bias_hh_l0.data, 0.0)
#         hidden_size = module.bias_hh_l0.data.shape[0]
#         module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0
# # 
#         if (module.bidirectional):
#             nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
#             nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
#             nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
#             nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
#             module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
# # # 
# #     elif isinstance(module, nn.LSTM):
#         nn.init.xavier_uniform_(module.weight_ih_l0.data)
#         nn.init.orthogonal_(module.weight_hh_l0.data)
#         nn.init.constant_(module.bias_ih_l0.data, 0.0)
#         nn.init.constant_(module.bias_hh_l0.data, 0.0)
#         hidden_size = module.bias_hh_l0.data.shape[0] // 4
#         module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0
# # 
#         if (module.bidirectional):
#             nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
#             nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
#             nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
#             nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
#             module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
# 

