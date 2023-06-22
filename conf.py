"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

import torch
import sentencepiece as spm

bpe_flag= True
flag_distill= False

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers=1#0
shuffle=True

# model parameter setting
batch_size = 24
max_len = 2000
d_model = 256
n_encoder_layers=2
n_decoder_layers=6
n_heads = 8
n_enc_replay = 6
dim_feed_forward= 2048
drop_prob = 0.1
depthwise_kernel_size=31
max_utterance_length= 360 #max nummber of labels in training utterances

lstm_hidden_size = 256
num_lstm_layers = 1

src_pad_idx=0
trg_pad_idx=30
trg_sos_idx=1
trg_eos_idx=31
enc_voc_size=32
dec_voc_size=32

lexicon="lexicon.txt"
tokens="tokens.txt"

sp = spm.SentencePieceProcessor()
if bpe_flag == True:
    sp.load('sentencepiece/build/libri.bpe-256.model')
    src_pad_idx=0
    trg_pad_idx=126
    trg_sos_idx=1
    trg_eos_idx=2
    enc_voc_size=sp.get_piece_size()
    dec_voc_size=sp.get_piece_size()
    lexicon="sentencepiece/build/librispeech-bpe-256.lex"
    tokens="sentencepiece/build/librispeech-bpe-256.tok"

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

args = {"src_pad_idx":src_pad_idx,
        "n_enc_replay":n_enc_replay,
        "d_model":d_model,
        "enc_voc_size":enc_voc_size,
        "dec_voc_size":dec_voc_size,
        "lstm_hidden_size":lstm_hidden_size,
        "num_lstm_layers":num_lstm_layers,
        "max_len":max_len,
        "dim_feed_forward":dim_feed_forward,
        "n_head":n_heads,
        "n_encoder_layers":n_encoder_layers,
        "features_length":n_mels,
        "drop_prob":drop_prob,
        "depthwise_kernel_size":depthwise_kernel_size,
        "device":device,
        "log_interval":100,
}