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
from torch.optim import Adam,AdamW

import sys

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.data_loader import text_transform
from conf import *

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')



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


model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
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
print("batch_size:",batch_size," num_heads:",n_heads," num_encoder_layers:", n_encoder_layers," num_decoder_layers:", n_decoder_layers, " optimizer:","NOAM[warmup ",warmup, "]","vocab_size:",dec_voc_size,"SOS,EOS,PAD",trg_sos_idx,trg_eos_idx,trg_pad_idx,"data_loader_len:",len(data_loader),"DEVICE:",device) 

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

#optimizer = NoamOpt(d_model, warmup, AdamW(params=model.parameters(),
#                                          lr=0, betas=(0.9, 0.98), eps=adam_eps, weight_decay=weight_decay))

optimizer = NoamOpt(d_model, warmup, Adam(params=model.parameters(),
                                          lr=0, betas=(0.9, 0.98), eps=adam_eps)) 
    
def train(iterator):
    
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        if not batch:
            continue
        
        src = batch[0].to(device) 
        trg = batch[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
        trg_expect =batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
        print("EXPECTED]:",text_transform.int_to_text(trg_expect[0]))
        #print("INPUT:",text_transform.int_to_text(trg[0]))

        output, encoder = model(src, trg)

        '''
        i_stripped=torch.IntTensor(text_transform.text_to_int(text_transform.int_to_text(trg[0]).replace("#",""))).to(device)
        out,scores,best_combined=beam_search(model, src[0].unsqueeze(0),  i_stripped.unsqueeze(0), max_length=max_utterance_length, beam_size=1, return_best_beam=True, weight_ctc=0.5)
        best_combined=torch.IntTensor(best_combined)
        print(" BEAM_OUT:",  text_transform.int_to_text(best_combined.squeeze(0)))
        '''                               
        
        loss_ce = loss_fn(output.permute(0,2,1), trg_expect)
        
        ctc_input_len=torch.full(size=(encoder.size(0),), fill_value = encoder.size(1), dtype=torch.long)
        ctc_target_len=batch[2]
        loss_ctc = ctc_loss(encoder.permute(1,0,2),batch[1],ctc_input_len,ctc_target_len).to(device)

        loss = 0.7 * loss_ce + 0.3 * loss_ctc

        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        out_text = output.data.topk(1)[1]
        print("OUT:",text_transform.int_to_text(out_text[0]))   
        

        epoch_loss += loss.item()

        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item(), '  loss_ce:', loss_ce.item(), '  loss_ctc:', loss_ctc.item())

    return epoch_loss / len(iterator)


def validate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    prev_loss = 9999999
    nepoch = 19#-1
    moddir=os.getcwd()+'/trained_model/seq_to_seq/'
    os.makedirs(moddir, exist_ok=True)            
            
    best_model=moddir+'{}mod{:03d}-transformer'.format('',nepoch)
    best_lr=moddir+'{}lr{:03d}-transformer'.format('',nepoch)

    if os.path.exists(best_model):
        print('loading model checkpoint:',best_model)
        model.load_state_dict(torch.load(best_model,map_location=device))
    if os.path.exists(best_lr):
        print('loading learning rate checkpoint:',best_lr)
        optimizer.load_state_dict(torch.load(best_lr))         

    for step in range(nepoch + 1, total_epoch):
        start_time = time.time()
        #for data in data_loader:
        #    print(data[1])
        #sys.exit()
            
        total_loss = train(data_loader)

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


        
        '''
        valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')
        '''


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  
    run(total_epoch=epoch, best_loss=inf)
