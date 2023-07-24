"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import argparse

from torch import nn, optim
import os
import torchaudio

from torchaudio.models.decoder import ctc_decoder
import sys
import re

from models.model.early_exit import *

from util.epoch_timer import epoch_time
from util.data_loader import text_transform, collate_infer_fn
from util.tokenizer import *
from util.beam_infer import *

from conf import *

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="Name of model dir in model_results (e.g., 'bi-lstm')")
parser.add_argument("model_type", type = str, help = "Type of model to train ('Early_Conformer' or 'Early_LSTM_Conformer')")
parser.add_argument("avg_model_start", type=int, help="Start of range for averaging models")
parser.add_argument("avg_model_end", type=int, help="End of range for averaging models")
args = parser.parse_args()

if (args.model_type == 'Early_Conformer'):
    model = Early_Conformer(src_pad_idx = src_pad_idx,
                                n_enc_replay = n_enc_replay,
                                d_model = d_model,
                                enc_voc_size = enc_voc_size,
                                dec_voc_size = dec_voc_size,
                                max_len = max_len,
                                dim_feed_forward = dim_feed_forward,
                                n_head = n_heads,
                                n_encoder_layers = n_encoder_layers,
                                features_length = n_mels,
                                drop_prob = drop_prob,
                                depthwise_kernel_size = depthwise_kernel_size,
                                device = device).to(device)  
elif (args.model_type == 'Early_LSTM_Conformer'):
    model = Early_LSTM_Conformer(src_pad_idx = src_pad_idx,
                                n_enc_replay = n_enc_replay,
                                d_model = d_model,
                                enc_voc_size = enc_voc_size,
                                dec_voc_size = dec_voc_size,
                                lstm_hidden_size = lstm_hidden_size,
                                num_lstm_layers = num_lstm_layers,
                                max_len = max_len,
                                dim_feed_forward = dim_feed_forward,
                                n_head = n_heads,
                                n_encoder_layers = n_encoder_layers,
                                features_length = n_mels,
                                drop_prob = drop_prob,
                                depthwise_kernel_size = depthwise_kernel_size,
                                device = device).to(device)  
elif(args.model_type == 'Early_Sequence_Conformer'):
    model = Early_Sequence_Conformer(src_pad_idx = src_pad_idx,
                                n_enc_replay = n_enc_replay,
                                trg_pad_idx = trg_pad_idx,
                                d_model = d_model,
                                enc_voc_size = enc_voc_size,
                                dec_voc_size = dec_voc_size,
                                max_len = max_len,
                                dim_feed_forward = dim_feed_forward,
                                n_head = n_heads,
                                n_encoder_layers = n_encoder_layers,
                                n_decoder_layers = n_decoder_layers,
                                features_length = n_mels,
                                drop_prob = drop_prob,
                                depthwise_kernel_size = depthwise_kernel_size,
                                device = device).to(device)                                                                                                                                                             

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
print("batch_size:",batch_size," num_heads:",n_heads," num_encoder_layers:", n_encoder_layers,"vocab_size:",dec_voc_size,"DEVICE:",device) 

loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)

def evaluate(model):

    file_dict='librispeech.lex'
    words=load_dict(file_dict)

    path = os.getcwd()+'/trained_model/' + args.model_name + '/'
    
    outpath = os.getcwd()+'/model_results/' + args.model_name + '_' + str(args.avg_model_start) + '_' + str(args.avg_model_end) + '/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    avg_model_start = args.avg_model_start
    avg_model_end = args.avg_model_end

    model = avg_models(model, path, avg_model_start, avg_model_end) 

    model.eval()
    #w_ctc = float(sys.argv[1])

    beam_size=10
    batch_size=25
    for set_ in ["test-clean"]: #["test-clean","test-other","dev-clean","dev-other"]:
        print(set_)
        
        test_dataset = torchaudio.datasets.LIBRISPEECH("/workspace", url=set_, download=False)
        data_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_infer_fn)

        with open(outpath + set_ + '.txt', 'w') as ofile:
            preds = [[], [], [], [], [], []]

            for batch in data_loader: 
                trg_expect =batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
                trg = batch[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
                # for trg_expect_ in trg_expect:
                #     if bpe_flag == True:
                #         outfile.write(set_ + " EXPECTED: " + sp.decode(trg_expect_.squeeze(0).tolist()).lower() + '\n')
                #     else:                    
                #         outfile.write(set_ + " EXPECTED: " + re.sub(r"[#^$]+","",text_transform.int_to_text(trg_expect_.squeeze(0))) + '\n')
                valid_lengths=batch[2]

                encoder=model(batch[0].to(device), valid_lengths)
                i=0
                
                for enc in encoder:
                    i = i + 1
                    best_combined = ctc_predict(enc, i - 1)
                    for best_ in best_combined:
                        if bpe_flag==True:
                            preds[i-1].append(apply_lex(sp.decode(best_).lower(),words) + '\n')
                            ofile.write(set_ + " BEAM_OUT_" + str(i) + ": " + apply_lex(sp.decode(best_).lower(),words) + '\n')
                        else:
                            preds[i-1].append(apply_lex(re.sub(r"[#^$]+","",best_.lower()),words) + '\n')
                            ofile.write(set_ + " BEAM_OUT_" + str(i) + ": " + apply_lex(re.sub(r"[#^$]+","",best_.lower()),words) + '\n')  

                print('Batch complete')  

        for i in range(6):
            outfile_name = outpath + set_ + '/layer' + str((i+1)*2) + '.txt'
            os.makedirs(os.path.dirname(outfile_name), exist_ok=True)    
            with open(outfile_name, 'w') as outfile: 
                outfile.writelines(preds[i])
                          
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn') 
    evaluate(model)
