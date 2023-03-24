"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers=10
shuffle=True

# model parameter setting
batch_size = 16
max_len = 2000
d_model = 256
n_encoder_layers=12
n_decoder_layers=6
n_heads = 8
dim_feed_forward= 2048
drop_prob = 0.1
max_utterance_length= 360 #max nummber of labels in training utterances

src_pad_idx=0
trg_pad_idx=30
trg_sos_idx=1
trg_eos_idx=31
enc_voc_size=32
dec_voc_size=32

bpe_flag= False
sample_rate = 16000
n_fft = 512
win_length = 320 #20ms
hop_length = 160 #10ms
n_mels = 80
n_mfcc = 80


# optimizer parameter setting
init_lr = 1e-5
factor = 0.9
adam_eps = 1e-9#5e-9
patience = 10
warmup = 8000 #dataloader.size()
epoch = 10000
clip = 1.0
weight_decay = 5e-4
#weight_decay = 0.1 # pytorch transformer class 
inf = float('inf')
