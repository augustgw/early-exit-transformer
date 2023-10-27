"""
@Author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import torch

from torch import nn, optim
import os
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from torch.optim import Adam,AdamW

import sys

from models.model.early_exit import Early_encoder, Early_transformer, Early_conformer, full_conformer

from util.data_loader import text_transform

from util.data_loader import pad_sequence
from util.beam_infer import ctc_predict_, ctc_cuda_predict
from util.beam_infer import greedy_decoder
from util.data_loader import collate_padding_fn
from conf import *
from data import *

#from voxpopuliloader import VOXPOPULI

torch.set_num_threads(10) 
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


train_dataset1 = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="train-clean-100", download=False)
train_dataset2 = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="train-clean-360", download=False)
train_dataset3 = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="train-other-500", download=False)
train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2,train_dataset3])
data_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_padding_fn, num_workers=num_workers)
#data_loader_1 = torch.utils.data.DataLoader(train_voxpopuli_50, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers) 

#data_loader = torch.utils.data.DataLoader(train_tedlium, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)


#data_loader = torch.utils.data.DataLoader(train_voxpopuli, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
            from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        print("RATE:",rate)
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))) 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

#n_enc_replay=6
#n_encoder_layers=2
d_model=256
n_heads=8
dim_feed_forward=2048

sp.load('sentencepiece/build/librispeech.bpe-5000.model')
enc_voc_size=sp.get_piece_size()
dec_voc_size=sp.get_piece_size()
lexicon="sentencepiece/build/librispeech-bpe-5000.lex"
tokens="sentencepiece/build/librispeech-bpe-5000.tok"            

model = Early_conformer(src_pad_idx=src_pad_idx,
                        n_enc_replay=n_enc_replay,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        dim_feed_forward=dim_feed_forward,
                        n_head=n_heads,
                        n_encoder_layers=n_encoder_layers,
                        features_length=n_mels,
                        drop_prob=drop_prob,
                        depthwise_kernel_size=depthwise_kernel_size,
                        device=device).to(device)    

print(f'The model has {count_parameters(model):,} trainable parameters')
#print("batch_size:",batch_size," num_heads:",n_heads," num_encoder_layers:", n_encoder_layers," num_decoder_layers:", n_decoder_layers, " optimizer:","NOAM[warmup ",warmup, "]","vocab_size:",dec_voc_size,"SOS,EOS,PAD",trg_sos_idx,trg_eos_idx,trg_pad_idx,"data_loader_len:",len(data_loader),"DEVICE:",device)
#warmup=len(data_loader_1) * 30
warmup=len(data_loader) * n_batch_split
print("batch_size:",batch_size," num_heads:",n_heads," num_encoder_layers:", n_encoder_layers, " optimizer:","NOAM[warmup ",warmup, "]","vocab_size:",dec_voc_size,"SOS,EOS,PAD",trg_sos_idx,trg_eos_idx,trg_pad_idx,"data_loader_len:",len(data_loader),"DEVICE:",device) 

model.apply(initialize_weights)

'''
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)
'''

loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)

optimizer = NoamOpt(d_model, warmup, AdamW(params=model.parameters(),lr=0, betas=(0.9, 0.98), eps=adam_eps, weight_decay=weight_decay))

#optimizer = NoamOpt(d_model, warmup, Adam(params=model.parameters(),lr=0, betas=(0.9, 0.98), eps=adam_eps))


def train(iterator):
    
    model.train()
    epoch_loss = 0
    len_iterator = len(iterator)
    for i,c_batch in enumerate(iterator):
        if len(c_batch) != 4:
            continue

        for batch_0,batch_1,batch_2,batch_3 in c_batch:

            src = batch_0.to(device) 
            trg = batch_1[:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
            trg_expect =batch_1[:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   

            valid_lengths=batch_3
            encoder = model(src, valid_lengths)

            ctc_target_len=batch_2
            loss_ctc = 0
            #loss_ce = 0

            ctc_input_len=torch.full(size=(encoder.size(1),), fill_value = encoder.size(2), dtype=torch.long)
            #print(encoder.size(),ctc_input_len)

            for enc in encoder:
                loss_ctc += ctc_loss(enc.permute(1,0,2),batch_1,ctc_input_len,ctc_target_len).to(device)
            del encoder
        
            loss = loss_ctc 

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
        print('step :', round((i / len_iterator) * 100, 2), '% , loss :', loss.item())
        if i % 500 ==0:
            print("EXPECTED:",sp.decode(trg_expect[0].tolist()).lower())
            #print("CTC_OUT at [",i,"]:",sp.decode(ctc_predict_(enc[0].unsqueeze(0))).lower()) 
            best_combined = ctc_cuda_predict(enc[0].unsqueeze(0), 5, tokens)
            print("CTC_OUT at [",i,"]:",sp.decode(best_combined[0][0].tokens).lower())             

            
    return epoch_loss / len_iterator


def run(total_epoch, best_loss, data_loader):

    train_losses, test_losses, bleus = [], [], []
    prev_loss = 9999999
    nepoch = -1

    moddir=os.getcwd()+'/trained_model/bpe_ctc/'
    os.makedirs(moddir, exist_ok=True)            
    initialize_model=False
    best_model=moddir+'{}mod{:03d}-transformer'.format('',nepoch)   #OCIO PAY ATTENTION REMOVE!!!
    
    best_lr=moddir+'{}lr{:03d}-transformer'.format('',nepoch)
    
    if os.path.exists(best_model):
        initialize_model=False
        print('loading model checkpoint:',best_model)
        model.load_state_dict(torch.load(best_model,map_location=device))

    if os.path.exists(best_lr):
        print('loading learning rate checkpoint:',best_lr)
        optimizer.load_state_dict(torch.load(best_lr))

    if initialize_model == True:
        total_loss=0
        for step in range(0, 30):
            print("Initializing step:",step)
            total_loss+=train(data_loader_1)
            print("TOTAL_LOSS-",step,":=",total_loss)

    for step in range(nepoch + 1, total_epoch):
        start_time = time.time()
        #for data in data_loader:
        #    print(data[1])
        #sys.exit()

        total_loss=train(data_loader)
        print("TOTAL_LOSS-",step,":=",total_loss)
        
        thr_l = (prev_loss - total_loss) / total_loss
        if total_loss < prev_loss:
            prev_loss = total_loss
            best_model=moddir+'mod{:03d}-transformer'.format(step)
            
            print("saving:",best_model)
            torch.save(model.state_dict(), best_model)
            lrate=moddir+'lr{:03d}-transformer'.format(step)            
            print("saving:",lrate)
            torch.save(optimizer.state_dict(), lrate)
        else:
            worst_model=moddir+'mod{:03d}-transformer'.format(step)
            print("WORST: not saving:",worst_model)            
        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  
    run(total_epoch=epoch, best_loss=inf, data_loader=data_loader)
