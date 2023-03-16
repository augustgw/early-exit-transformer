#!/datadrive/speechtek/daniele/anaconda3/bin/python
## Usage

import torch

import torch.nn as nn
from torch import Tensor
import copy
from typing import Optional, Any, Union, Callable
import numpy as np
import random
import sys
import io
#import editdistance
import re

#from torchtext.vocab import vocab
import math
import os
import torchaudio
from torch.optim import Adam
from torch.autograd import Variable
from typing import Tuple
from torch.utils import data

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer


cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

torch.set_num_threads(10)

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


class Conv1dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)
    Args:
        in_channels (int): Number of channels in the input image                                             
        out_channels (int): Number of channels produced by the convolution                       
                                                                                                                
    Inputs: inputs                                                       
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs
    Returns: outputs, output_lengths 
        - **outputs** (batch, time, dim): Tensor produced by the convolution

   """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros'),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=0, padding_mode='zeros')
        )
    def forward(self, inputs: Tensor) -> torch.tensor:
        outputs = self.sequential(inputs)
        return outputs

class _PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(1), :].squeeze(1).repeat(token_embedding.size(0),1,1)) 
        
#        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class PositionalEncoding(nn.Module):

    def __init__(self,  d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x=x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2)
        return self.dropout(x)

        
class MyTransformer(nn.Module):
    def __init__(
            self,
            d_model: int = 512,
            norm_first: bool = False,
            features_length = 80,
            num_heads: int = 8,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 3,
            dim_feedforward = 1024,
            n_phones: int = 32,
            dropout_f: float = 0.1,
            batch_first: bool = True,
      ) -> None:
#        super(MyTransformer, self).__init__()
        super().__init__()
        self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=dropout_f, max_len=2000)#20000)
        self.layer_norm=nn.LayerNorm(d_model,eps=1e-5) 
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_f, norm_first=norm_first, batch_first=batch_first),  num_encoder_layers,self.layer_norm)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout_f, batch_first=batch_first, norm_first = norm_first),  num_decoder_layers, self.layer_norm) 
#        self.my_model= nn.Transformer(dropout=dropout_f, norm_first=norm_first, d_model=d_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=batch_first)
        self.generate_square_subsequent_mask = nn.Transformer.generate_square_subsequent_mask 
        self.linear=nn.Linear(d_model, n_phones)
        self.linear=nn.Linear(d_model, n_phones)
        self.softmax=nn.Softmax(dim=2)
        self.relu=nn.ReLU()
        self.embedding = nn.Embedding(n_phones, d_model) 

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
            # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
            # [False, False, False, True, True, True]
            return (matrix == pad_token)
        
    def create_tgt_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
    def _encoder_(self, src: Tensor, src_key_padding_mask: Optional[Tensor] = None ) -> Tensor:
        #src = self.conv_subsample(src.permute(0,2,1)).permute(0,2,1)

        src = src * math.sqrt(d_model)
        src = self.positional_encoder(src)

        encoder_out=self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return encoder_out        

    def _decoder_(self, tgt: Tensor, encoder_out: Tensor, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt = self.embedding(tgt) * math.sqrt(d_model)
        tgt = self.positional_encoder(tgt)
        decoder_out=self.decoder(tgt,encoder_out,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        out=self.linear(decoder_out)         

        return out        
    def _conv_subsample_(self, src: Tensor) -> Tensor:
        return self.conv_subsample(src).permute(0,2,1)
    
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:


        # Src size must be (batch_size, src sequence length)
        
        # Tgt size must be (batch_size, tgt sequence length)

        #print("SRCLEN_:",src.size())
        #self.conv_subsample = Conv1dSubampling(in_channels=features_length, out_channels=d_model)
        src = self.conv_subsample(src.permute(0,2,1)).permute(0,2,1)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)

        #src = self.embedding(src) * math.sqrt(d_model)
        src = src * math.sqrt(d_model)
        tgt = self.embedding(tgt) * math.sqrt(d_model)
        
        tgt = self.positional_encoder(tgt)
        src = self.positional_encoder(src)
        #print("TGT_MASK:",tgt_mask)
        #print("TGT_KEY_MASK:",tgt_key_padding_mask)


        encoder_out=self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        decoder_out=self.decoder(tgt,encoder_out,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        decoder_out=self.linear(decoder_out)
        decoder_out=torch.nn.functional.log_softmax(decoder_out,dim=2)        

        encoder_out=self.linear(encoder_out)
        encoder_out=torch.nn.functional.log_softmax(encoder_out,dim=2)

        return decoder_out, encoder_out



# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            # compute the accuracy over all test images
            accuracy = (100 * accuracy / total)
            return(accuracy)

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
        # 30
        ^ 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        ' 29
        $ 31
        @ 0
        """
# ^=<SOS> 1
# $=<EOS> 31
# #=<PAD> 30
# @=<blank> for ctc
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()

            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

        self.index_map[28] = ' '
        
    def text_to_int(self,text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = 28#self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i.detach().item()])
        return ''.join(string)#.replace('', ' ')
    
sample_rate = 16000
n_fft = 512
win_length = 320 #20ms
hop_length = 160 #10ms
n_mels = 80
n_mfcc = 80

dim_feedforward=2048#1024

torch.set_printoptions(profile="full")
import torchaudio.transforms as T

mfcc_transform = T.MFCC(sample_rate=sample_rate, \
                        n_mfcc=n_mfcc, \
                        melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'win_length': win_length, 'hop_length': hop_length})

spec_transform = T.Spectrogram(n_fft=n_fft * 2, \
                               hop_length=hop_length, \
                               win_length=win_length)

melspec_transform = T.MelScale(sample_rate=sample_rate, n_mels=n_mels, n_stft=n_fft+1)

freq_masking=T.FrequencyMasking(freq_mask_param=80)
time_masking=T.TimeMasking(time_mask_param=80)    


n_phones=32
d_model=256
num_heads=8
num_encoder_layers=12
num_decoder_layers=6
model = MyTransformer(n_phones=n_phones, \
                      d_model=d_model, \
                      features_length=n_mfcc, \
                      num_heads=num_heads, \
                      norm_first=True, \
                      num_encoder_layers=num_encoder_layers, \
                      dim_feedforward=dim_feedforward, \
                      num_decoder_layers=num_decoder_layers, \
                      batch_first=True).to(device)

loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)

learning_rate=0.008 #google
weight_decay=0.0001
#opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

text_transform=TextTransform()

def pad_sequence(batch, padvalue):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=padvalue)
    return batch.permute(0, 2, 1)


def ins_eos(inpseq, EOS_token=31):
    targets = torch.LongTensor(inpseq.size(0),1,inpseq.size(2)+1)
    for k in range(0,inpseq.size(0)):
        targets[k][0][0:inpseq.size(2)]=inpseq[k][0]
        targets[k][0][inpseq.size(2)]=EOS_token
    return targets      

def collate_fn(batch, SOS_token=1, EOS_token=31, PAD_token=30):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets, tensors_f, tensors_t = [], [], [], []

    # Gather in lists, and encode labels as indices
    #t_len=torch.LongTensor(len(batch) * 3,1)
    t_len=[]

    for waveform, _, label, spkid, _, _,  in batch: #spkid, cha_id, ut_id in batch:
        if len(label) < 351:
            #spec=mfcc_transform(waveform).to(device)
            spec=spec_transform(waveform)#.to(device)
            #spec_1 = freq_masking(spec)#.to(device)
            #spec_1 = time_masking(spec_1)#.to(device)

            #spec_1 = melspec_transform(spec_1).to(device)
            spec = melspec_transform(spec).to(device)
            #print('SPEC12:',torch.eq(torch.tensor(spec_1),torch.tensor(spec_2)))



            #pad_spec=torch.zeros(spec.size(0),spec.size(1), 3600 - spec.size(2)).to(device)
            #spec=torch.cat((spec,pad_spec),2)
            #del pad_spec
            #tensors_f += spec_1#[waveform]
            tensors += spec#[waveform]
            #del spec_1
            del spec
            
            #label=label[10:110]
            #tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()))
            tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()+"$"))
            #pad_tg=torch.full((620 - tg.size(0),),PAD_token)
            #tg=torch.cat((tg,pad_tg),0)
            #del pad_tg                   
        
            targets += [tg.unsqueeze(0)]
            lb=[len(tg)]

            t_len+=lb
            #t_len+=lb

            del waveform
            del label
        else:
            print('REMOVED:',spkid,' LAB:',label)
    # Group the list of tensors into a batched tensor
    if tensors:
        #tensors=tensors + tensors_f
        #targets=targets + targets 
        #temp=list(zip(tensors,targets))
        #random.shuffle(temp)
        #r1,r2=zip(*temp)
        #tensors,targets=list(r1),list(r2)
        #del temp
        tensors = pad_sequence(tensors,0)
        targets = pad_sequence(targets,PAD_token)
        #targets = ins_eos(targets, EOS_token=EOS_token)
        
        #del tensors_f

        return tensors.squeeze(1), targets.squeeze(1), torch.tensor(t_len)
    else:
        return None

def _collate_fn_(batch, SOS_token=1, EOS_token=31, PAD_token=30):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets, source_pad = [], [], []

    # Gather in lists, and encode labels as indices
    t_len=torch.LongTensor(len(batch),1)
    k=0
    for waveform, _, label, spkid, _, _,  in batch: #spkid, cha_id, ut_id in batch:
        spec=mfcc_transform(waveform).to(device)
        #spec_f = freq_masking(spec).to(device)
        #spec_t = time_masking(spec).to(device)
        #pad_spec=torch.zeros(spec.size(0),spec.size(1), 3600 - spec.size(2)).to(device)
        #spec=torch.cat((spec,pad_spec),2)
        #del pad_spec
        tensors += spec#[waveform]
        #del spec

        #label=label[10:110]
        #tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()))
        tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()+"$"))
        #pad_tg=torch.full((620 - tg.size(0),),PAD_token)
        #tg=torch.cat((tg,pad_tg),0)
        #del pad_tg                   


        
        targets += [tg.unsqueeze(0)]
        t_len[k][0]=len(tg)
        k=k+1
        del waveform
        del label

    # Group the list of tensors into a batched tensor

    tensors = pad_sequence(tensors,0)
    targets = pad_sequence(targets,PAD_token)
    #targets = ins_eos(targets, EOS_token=EOS_token)
    return tensors.squeeze(1), targets.squeeze(1), t_len


def predict(model, input_sequence, max_length=80, SOS_token=1, EOS_token=31):
        """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
        y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

        encoder_out = model._encoder_(input_sequence)

        for _ in range(max_length):
            # Get source mask


            tgt_mask=model.create_tgt_mask(y_input.size(1)).to(device)
            tgt_pad_mask=model.create_pad_mask(y_input,30).to(device) 

            pred = model._decoder_(y_input, encoder_out = encoder_out, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_pad_mask)
            del tgt_mask
            del tgt_pad_mask
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = torch.tensor([[next_item]], device=device)

            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)
            
            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_token:
#                print("DETECT EOS")
                break
            del next_item
            del pred

#        print("Y_OUT:",text_transform.int_to_text(y_input.squeeze(0)).replace('#',''))
        del encoder_out
        return y_input#.view(-1).tolist()


def adjust_learning_rate(opt, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in opt.param_groups:
        param_group['lr'] = lr                                                                               
def load_dict(file_path):
    dict=[]
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            dict += [line.strip("\n")]
    return dict
def apply_lex(predicted, lexicon):
    lex_out=[]
    for w in predicted.split(" "):
        if w in lexicon:
            lex_out += [w]
        else:
            min_lex=99999
            w_min=""
            for w_lex in lexicon:
                d_lex=editdistance.eval(w, w_lex)
                if d_lex < min_lex:
                    min_lex = d_lex
                    w_min = w_lex
            lex_out += [w_min]
                    
    return " ".join([str(item) for item in lex_out])
                
def train(num_epochs):
    global learning_rate
    batch_size=4#24#16
    prev_loss=99999999
    n_worst=0
    
#    test_dataset = torchaudio.datasets.LIBRISPEECH("/data/corpora/", url="test-clean", download=False)
    train_dataset1 = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="train-clean-100", download=False)
    train_dataset2 = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="train-clean-360", download=False)
    train_dataset3 = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="train-other-500", download=False)        

#    train_dataset1 = torchaudio.datasets.LIBRISPEECH("/data/falavi/transformer.3/libri_second", url="train-clean-100", download=False)
#    train_dataset2 = torchaudio.datasets.LIBRISPEECH("/data/falavi/transformer.3/libri_second", url="train-clean-360", download=False)
#    train_dataset3 = torchaudio.datasets.LIBRISPEECH("/data/falavi/transformer.3/libri_second", url="train-other-500", download=False)        
    
    train_dataset = train_dataset1#torch.utils.data.ConcatDataset([train_dataset1,train_dataset2,train_dataset3])

    data_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=False, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    nepoch=0
    #best_model='{}/model/first_nheads{:02d}_nenclayers{:02d}/mod{:03d}-transformer'.format('.',num_heads,num_encoder_layers,nepoch)

    
    #best_model='{}/model/all-adamW/mod{:03d}-transformer'.format('.',nepoch)        

    #model.load_state_dict(torch.load(best_model))
    
    #m1=model.state_dict()

    '''
    nc = 1
    for nepoch in range(nepoch+1,87):
#        best_model='{}/model/first_nheads{:02d}_nenclayers{:02d}/mod{:03d}-transformer'.format('.',num_heads,num_encoder_layers,nepoch)
        best_model='{}/model/all-adamW/mod{:03d}-transformer'.format('.',nepoch)        
        if os.path.exists(best_model):
            print("Averaging with:", best_model) 
            model.load_state_dict(torch.load(best_model))
            m2=model.state_dict()
            for key in m2:
                m1[key] = m2[key]+m1[key]
            nc = nc +1
    del m2
    for key in m1:
        m1[key] = m1[key] / nc
    '''
    #model.load_state_dict(m1)      
    #del m1
    
    #best_lr='{}/model/first_nheads{:02d}_nenclayers{:02d}/lr{:03d}-transformer'.format('.',num_heads,num_encoder_layers,nepoch)
    #best_lr='{}/model/all-adamW/lr{:03d}-transformer'.format('.',nepoch)

    augment_factor=1
    warmup=16#len(data_loader) * augment_factor  #around the size of teh dataloader
    #opt = NoamOpt(d_model, warmup, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    opt = NoamOpt(d_model, warmup, torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.1))

    #opt.load_state_dict(torch.load(best_lr))

    '''
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    '''     
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print("TOT_PARAMS:",total_params)    
    
    iprint=0
    model.train()

    print("batch_size:",batch_size," num_heads:",num_heads," num_encoder_layers:", num_encoder_layers," num_decoder_layers:", num_decoder_layers, " optimizer:","NOAM[warmup ",warmup, "]","DEVICE:",device)

    print("LENGTH OF DATA_LOADER:",len(data_loader)*augment_factor)
    for epoch in range(nepoch+1,num_epochs):  # loop over the dataset multiple times
        total_loss=0
        for data in data_loader:
            if not data:
                continue
            y_input = data[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
            #print("Y_INPUT:",y_input)

            y_expected = data[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]

            print("EXPECTED[",iprint,"]:",text_transform.int_to_text(y_expected[iprint]))
            print("INPUT[",iprint,"]:",text_transform.int_to_text(y_input[iprint]))            

            #spec=mfcc_transform(data[0]).transpose(1,2).to(device)
            #print(data[0].size())
            spec=data[0].transpose(1,2).to(device)            

            tgt_mask=model.create_tgt_mask(y_input.size(1)).to(device) #generate_square_subsequent_mask(sz).to(device)            
            tgt_pad_mask=model.create_pad_mask(y_input,30).to(device)
            print(tgt_pad_mask)
            print(tgt_mask)            
            #print(data[2].size())
            #print(tgt_mask.size())
            #print(tgt_mask)            
            pred, enc_out = model(spec, y_input,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_pad_mask)
            #print(pred.size(),enc_out.size())
            ###pred, enc_out = model(spec, y_input,tgt_mask=tgt_mask)

            out_t = pred.data.topk(1)[1]

            print("OUT[",iprint,"]:",text_transform.int_to_text(out_t[iprint]))  

            loss_ce = loss_fn(pred.permute(0,2,1), y_expected).to(device)
            ctc_input_len=torch.full(size=(enc_out.size(0),), fill_value = enc_out.size(1), dtype=torch.long)
            ctc_target_len=data[2]

            #print(data[1],ctc_input_len,ctc_target_len)

            loss_ctc = loss_ce#ctc_loss(enc_out.permute(1,0,2),data[1],ctc_input_len,ctc_target_len).to(device)

            loss = loss_ce#0.7 * loss_ce + 0.3 * loss_ctc
            total_loss+= loss.detach().item()
            print("LOSS_CE:",loss_ce.detach().item()," LOSS_CTC:",loss_ctc.detach().item()," LOSS:",loss.detach().item())

            #opt.zero_grad()
            model.zero_grad()            
            #loss.backward()
            loss.backward()
            opt.step()
                                       
            del pred
            del loss_ce
            del loss_ctc
            del loss
            del ctc_input_len
            del ctc_target_len
            del spec

            del tgt_mask
            del y_input
            del y_expected
            del out_t
            del data
            del enc_out
            del tgt_pad_mask
            break
        print("TOTAL_LOSS-",epoch,":=",total_loss/(len(data_loader) * augment_factor))

        thr_l = (prev_loss - total_loss) / total_loss
        #moddir='{}/model/second_nheads{:02d}_nenclayers{:02d}'.format('.',num_heads,num_encoder_layers)
        moddir=os.getcwd()+'/model/all-adamW/'
        os.makedirs(moddir, exist_ok=True)            
        if total_loss < prev_loss:
            prev_loss = total_loss
            #best_model='{}/model/second_nheads{:02d}_nenclayers{:02d}/mod{:03d}-transformer'.format('.',num_heads,num_encoder_layers,epoch)
            best_model=moddir+'mod{:03d}-transformer'.format(epoch)
            
            print("saving:",best_model)
            #torch.save(model.state_dict(), best_model)
            #lrate='{}/model/second_nheads{:02d}_nenclayers{:02d}/lr{:03d}-transformer'.format('.',num_heads,num_encoder_layers,epoch)
            lrate=moddir+'lr{:03d}-transformer'.format(epoch)            
            #print("saving:",lrate)
            #torch.save(opt.state_dict(), lrate)
        else:
            #worst_model='{}/model/second_nheads{:02d}_nenclayers{:02d}/mod{:03d}-transformer'.format('.',num_heads,num_encoder_layers,epoch)
            worst_model=moddir+'mod{:03d}-transformer'.format(epoch)
                        
            print("WORST: not saving:",worst_model)            


def evaluate():
    batch_size=256
    file_dict='librispeech.lex'
    words=load_dict(file_dict)

    nepoch=52
    best_model='{}/model/best16/mod{:02d}-transformer'.format('.',nepoch)
    model.load_state_dict(torch.load(best_model))

    m1=model.state_dict()
    for nepoch in range(53,70):
        best_model='{}/model/best16/mod{:02d}-transformer'.format('.',nepoch)
        model.load_state_dict(torch.load(best_model))
        m2=model.state_dict()
        for key in m2:
            m1[key] = m2[key]+m1[key]
    for key in m1:
        m1[key] = m1[key] / 18
    model.load_state_dict(m1)
    
#    nepoch=47
#    best_model='{}/model/best16/mod{:02d}-transformer'.format('.',nepoch)
#    model.load_state_dict(torch.load(best_model))
    model.eval()
    for set_ in "test-clean",  "dev-clean", "test-other", "dev-other":

        test_dataset = torchaudio.datasets.LIBRISPEECH("/data/corpora/", url=set_, download=False)
    
        data_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn)

        for data in data_loader:
            print(set_,"SPKID", data[2], "UTID", data[3])
            y_expected = data[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            for y_expected_ in  y_expected:
                print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(y_expected_.squeeze(0))))
                del y_expected_

            spec=mfcc_transform(data[0]).transpose(1,2).to(device)

            for spec_ in spec:
                out=predict(model, spec_.unsqueeze(0), max_length=1200)    
                #print(set_,"OUT:",re.sub(r"[#^$]+","",text_transform.int_to_text(out.squeeze(0))))
                out_lex = apply_lex(re.sub(r"[#^$]+","",text_transform.int_to_text(out.squeeze(0))),words)
                print(set_,"OUT_LEX:",out_lex)
                
                del out
                del out_lex
                del spec_
                
            del y_expected
            del spec
            del data
# protect the entry point
if __name__ == '__main__':
        # set the start method
        torch.multiprocessing.set_start_method('spawn')
        train(5000)
#evaluate()
