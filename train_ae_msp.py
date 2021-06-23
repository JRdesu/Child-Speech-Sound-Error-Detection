from argparse import ArgumentParser, Namespace
import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from tqdm import tqdm 
from tensorboardX import SummaryWriter
from make_dataset import phone_Dataset, get_data_loader
from utils import *
from model import AE_MSP, init_AE_MSP
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
class Solver(object):
    def __init__(self, config, args):
        self.config = config
        print(config)

        self.args = args
        print(self.args)

        # self.logger = Logger(self.args.logdir)
        self.writer = SummaryWriter(self.args.logdir)
        self.get_data_loaders()
        self.model = AE_MSP(self.config)
        if args.current_epoch == 0:
            self.build_model()
            self.save_config()
        else:
            self.load_model()

    def save_model(self, epoch):
        # save model and discriminator and their optimizer
        torch.save(self.model.state_dict(), f'{self.args.store_model_path}/ae_{epoch}.ckpt')
        torch.save(self.opt.state_dict(), f'{self.args.store_model_path}/ae_{epoch}.opt')

    def save_config(self):
        store_dir = os.path.dirname(self.args.store_model_path)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        with open(f'{self.args.store_model_path}/ae.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}/ae.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        self.model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters())
        self.opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
        return

    def get_data_loaders(self):
        datadir = self.args.datadir
        train_set = self.args.train_set
        fea_path = os.path.join(datadir, f'{train_set}_fea.pkl')
        ali_ctm_path = os.path.join(datadir, f'{train_set}_ali.ctm')
        self.train_set = phone_Dataset(fea_path, ali_ctm_path, 
                                        config['data_loader']['frame_shift'], 
                                        config['data_loader']['cut_numframes'])
        self.train_loader = get_data_loader(self.train_set, 
                                            config['data_loader']['batch_size'])
        return

    def build_model(self):        
        self.model = init_AE_MSP(self.model)
        self.model.to(device)
        self.model.train()
        print(self.model)
        optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(self.model.parameters(),
            lr=optimizer['lr'], betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], weight_decay=optimizer['weight_decay'])
        print(self.opt)
        return

    def train(self, n_epochs, start_epoch):
        n_iters = len(self.train_loader)
        for epoch in range(start_epoch, n_epochs):
            total_sum = 0.0
            msp_sum = 0.0
            rec_sum = 0.0
            for data, num_frames, phn_id, _, _, attr_label in tqdm(iter(self.train_loader)):
                data = cc(data)
                num_frames = cc(num_frames)
                attr_label = cc(attr_label)
                rec_out, z = self.model(data, num_frames)
                loss_total, loss_rec, loss_msp = self.model.loss(data, rec_out, z, attr_label, config['optimizer']['msp_weight'])
                total_sum += loss_total.item()
                msp_sum += loss_msp
                rec_sum += loss_rec
                self.opt.zero_grad()
                loss_total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                    max_norm=self.config['optimizer']['grad_norm'])
                self.opt.step()
                
            # ADD TO LOGGER  
            total_sum = total_sum/n_iters
            msp_sum = msp_sum/n_iters
            rec_sum = rec_sum/n_iters
            print(f'AE_MSP:[{epoch}], loss_rec={rec_sum:.2f}, loss_msp={msp_sum:.2f}, loss_total={total_sum:.2f}')             
            self.writer.add_scalar('loss/total', total_sum, epoch)
            self.writer.add_scalar('loss/rec', rec_sum, epoch)
            self.writer.add_scalar('loss/msp', msp_sum, epoch)            

            if epoch == 50:
                self.save_model(epoch)
            if epoch == 100:
                self.save_model(epoch)
            if epoch%300 == 0 and epoch>500:                        
                self.save_model(epoch)           
        return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', default='config/config5.yaml')
    parser.add_argument('--datadir', '-d', 
            default='/home/jiarui/PYTHON/arti_embed/data/phone_segments')
    parser.add_argument('--train_set', default='cuchild_train')
    parser.add_argument('--logdir', default='log/')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_model_path', default='ae_20')
    parser.add_argument('--tag', '-t', default='ae_msp_6')
    parser.add_argument('--epochs', default=1510, type=int)
    parser.add_argument('--current_epoch', default=0, type=int)


    args = parser.parse_args()
    args.store_model_path = os.path.join('storage', args.tag)
    args.load_model_path = os.path.join('storage', args.tag, args.load_model_path)
    args.logdir = os.path.join('log', args.tag)

    if not os.path.exists(args.store_model_path):
        os.makedirs(args.store_model_path)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    
    # load config file 
    with open(args.config) as f:
        config = yaml.safe_load(f)

    solver = Solver(config=config, args=args)
    solver.train(args.epochs, args.current_epoch)

