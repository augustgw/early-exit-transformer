"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers=0
shuffle=False
seq_to_seq_flag= True
# model parameter setting
batch_size = 1
max_len = 2000
d_model = 256
n_encoder_layers=12
n_decoder_layers=6
n_heads = 8
dim_feed_forward= 2048
drop_prob = 0.1
max_utterance_length= 640 #max nummber of labels in training utterances

src_pad_idx=0
trg_pad_idx=30
trg_sos_idx=1
trg_eos_idx=31
enc_voc_size=32
dec_voc_size=32

bpe_flag = False
sample_rate = 16000
n_fft = 512
win_length = 320 #20ms
hop_length = 160 #10ms
n_mels = 80
n_mfcc = 80

inf = float('inf')
