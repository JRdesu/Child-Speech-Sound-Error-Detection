import torch
from utils.functions import PhoneErrorRate
from data.loader import collate
import pdb
from tqdm import tqdm

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
            raw_prob = (torch.cat([each_y for each_y in raw_prob], 1)[:,:maxlen_dec-1,:]).contiguous()  #[B, max_token_len-1, num_classes] since the start_token is skip
            pred = preprocessor.forced_decode(torch.max(raw_prob, 2)[1][0])
            att = (torch.cat([each_att for each_att in attn], 1)[:,:maxlen_dec-1,:]).contiguous()
            output[utt] = {}
            output[utt]['att'] = att
            output[utt]['pred'] = pred
            output[utt]['prob'] = prob
            output[utt]['truth'] = true_y
            # 
    return output


def trainIters(loader, preprocessor, encoder, decoder, 
    optimizer, criterion, tf_rate):
    device = encoder.device
    encoder.train()
    decoder.train()
    loss = 0.0
    per = 0.0
    # 
    for batch in tqdm(loader): 
        x, y, x_len, y_len, _ = collate(*batch)
        x = x.to(device); x_len = x_len.to(device)
        y = y.to(device); y_len = y_len.to(device)
        # 
        batch_size = x.size(0)
        maxlen_enc = min(y.size(1), decoder.maxlen_dec)
        # 
        batch_loss = 0
        optimizer.zero_grad()
        encoder_output = encoder(x, x_len)
        raw_prob, attn = decoder(encoder_output, x_len, ground_truth=y, 
            teacher_force_rate=tf_rate)
        # 
        pred_y = (torch.cat([each_y for each_y in raw_prob], 1)[:,:maxlen_dec-1,:]).contiguous()
        pred_y = pred_y.permute(0, 2, 1).contiguous() #[B, num_classes, seq_lenth] [B, seq_lenth] # 
        true_y = y[:, 1:maxlen_dec]
        # 
        batch_loss = criterion(pred_y, true_y)  # devide by batch_size*time_steps [B, num_classes, seq_lenth] [B, seq_lenth] #
        batch_per = PhoneErrorRate(torch.max(pred_y, 1)[1], y, preprocessor)
        loss = loss + batch_loss.item()
        per = per + batch_per
        batch_loss.backward()
        optimizer.step()
 
    loss = loss/len(loader) #len(loader) is num of batches
    per = per/len(loader)
    return loss, per

def devIters(loader, preprocessor, encoder, decoder, criterion, tf_rate):
    '''
    '''
    device = encoder.device
    encoder.eval()
    decoder.eval()
    loss = 0.0
    per = 0.0
    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, x_len, y_len, _ = collate(*batch)
            x = x.to(device); x_len = x_len.to(device)
            y = y.to(device); y_len = y_len.to(device)
            # 
            batch_size = x.size(0)
            maxlen_dec = min(y.size(1), decoder.maxlen_dec)
            # 
            batch_loss = 0
            encoder_output = encoder(x, x_len)
            raw_prob, attn= decoder(encoder_output, x_len, ground_truth=None,
                teacher_force_rate=tf_rate)
            # 
            pred_y = (torch.cat([each_y for each_y in raw_prob], 1)[:,:maxlen_dec-1,:]).contiguous()
            pred_y = pred_y.permute(0, 2, 1) #[B, num_classes, maxlen_dec-1] since the start_token is skip
            true_y = y[:, 1:maxlen_dec].contiguous() #[B, maxlen_dec-1] 
            # 
            batch_loss = criterion(pred_y, true_y)  # devide by batch_size*time_steps
            batch_per = PhoneErrorRate(torch.max(pred_y, 1)[1], y, preprocessor)
            loss = loss + batch_loss.item()
            per = per + batch_per
             
        loss = loss/len(loader)
        per = per/len(loader)
    return loss, per

def aliIters(loader, preprocessor, encoder, decoder):
    '''
    bach_size is 1
    y: <s> h o i </s>
    y_len: 3 not 5
    '''
    device = encoder.device
    encoder.eval()
    decoder.eval()
    output = {}
    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, x_len, y_len, utt = collate(*batch)
            x = x.to(device); x_len = x_len.to(device)
            y = y.to(device); y_len = y_len.to(device)
            true_y = preprocessor.decode(y[0].contiguous()) #y:[B, y_len+1] 
            # 
            batch_size = x.size(0)
            # 
            encoder_output = encoder(x, x_len)
            raw_pred_prob, attention= decoder(encoder_output, x_len, ground_truth=None)
            # 
            pred_y = (torch.cat([each_y.unsqueeze(1) for each_y in raw_pred_prob], 1)).contiguous()            
            pred_y = pred_y.permute(0, 2, 1).contiguous() #[B, num_classes, max_token_len-1] since the start_token is skip
            pred = preprocessor.decode(torch.max(pred_y, 1)[1][0])
            prob = pred_y[:,:,:len(pred)] 
            att = (torch.cat([each_att[0].unsqueeze(1) for each_att in attention], 1)[:,:len(pred),:]).contiguous()              
            # 
            output[utt] = {}
            output[utt]['att'] = att
            output[utt]['pred'] = pred
            output[utt]['prob'] = prob
            output[utt]['truth'] = true_y       
            # 
    return output




