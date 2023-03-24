"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
import os
import torchaudio

from torchaudio.models.decoder import ctc_decoder
import sys
import re

from data_infer import *
from models.model.early_exit import Early_encoder, Early_transformer 
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.data_loader import text_transform
from util.tokenizer import *
from util.beam_infer import *
from conf_infer import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = Early_transformer(src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_sos_idx=trg_sos_idx,
            n_enc_replay=n_enc_replay,
            d_model=d_model,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            dim_feed_forward=dim_feed_forward,
            n_head=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            features_length=n_mels,
            drop_prob=drop_prob,
            device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
print("batch_size:",batch_size," num_heads:",n_heads," num_encoder_layers:", n_encoder_layers," num_decoder_layers:", n_decoder_layers,"vocab_size:",dec_voc_size,"SOS,EOS,PAD",trg_sos_idx,trg_eos_idx,trg_pad_idx,"data_loader_len:",len(data_loader),"DEVICE:",device) 

loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
def evaluate(model):

    file_dict='librispeech.lex'
    words=load_dict(file_dict)

    path = os.getcwd()+'/trained_model/early_exit/'

    model = avg_models(model, path, 31, 38)        
    model.eval()
    w_ctc = float(sys.argv[1])


    set_ = 'test'
    beam_size=30
    for batch in data_loader: 
        trg_expect =batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
        trg = batch[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
        for trg_expect_ in trg_expect:
            if bpe_flag == True:
                print(set_,"EXPECTED:",sp.decode(trg_expect_.squeeze(0).tolist()).lower())
            else:                    
                print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(trg_expect_.squeeze(0))))

        for spec_ in batch[0]:

            if seq_to_seq_flag == True:
                out,scores,best_combined=beam_search(model, spec_.unsqueeze(0),  trg, max_length=max_utterance_length, beam_size=beam_size, return_best_beam=True, weight_ctc=w_ctc)
            else:
                output,encoder=model(spec_.unsqueeze(0).to(device),trg.to(device))
                i=0
                for _,enc in zip(output,encoder):
                    best_combined = ctc_predict(enc)
                    
                    if bpe_flag==True:
                        print(set_," BEAM_OUT_",i,":",  apply_lex(sp.decode(best_combined).lower(),words))
                    else:
                        if seq_to_seq_flag == True:
                            best_combined=torch.IntTensor(best_combined)
                            print(set_," BEAM_OUT:",  apply_lex(re.sub(r"[#^$]+","",text_transform.int_to_text(best_combined.squeeze(0)).lower()),words))
                        else:
                            print(set_," BEAM_OUT_",i,":",  apply_lex(re.sub(r"[#^$]+","",best_combined.lower()),words))
                    i=i+1
                          
if __name__ == '__main__':
    evaluate(model)
