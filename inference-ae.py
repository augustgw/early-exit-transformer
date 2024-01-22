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

from data import *

from models.model.early_exit import Early_encoder, Early_transformer, Early_conformer, full_conformer


from util.epoch_timer import epoch_time

from util.data_loader import text_transform, spec_transform, melspec_transform, pad_sequence
from util.tokenizer import *
from util.beam_infer import *
from util.conf import *

from util.data_loader import collate_padding_fn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


d_model = 144
n_heads = 4
d_feed_forward = 1024
n_dec_layers = 4

model = full_conformer(trg_pad_idx=trg_pad_idx,
                       n_enc_exits=n_enc_exits,
                       d_model=d_model,
                       enc_voc_size=enc_voc_size,
                       dec_voc_size=dec_voc_size,
                       max_len=max_len,
                       d_feed_forward=d_feed_forward,
                       n_head=n_heads,
                       n_enc_layers=n_enc_layers,
                       n_dec_layers=n_dec_layers,
                       features_length=n_mels,
                       drop_prob=drop_prob,
                       depthwise_kernel_size=depthwise_kernel_size,
                       device=device).to(device)


print(f'The model has {count_parameters(model):,} trainable parameters')
print("batch_size:", batch_size, " num_heads:", n_heads, " num_encoder_layers:",
      n_enc_layers, "vocab_size:", dec_voc_size, "DEVICE:", device)

loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def collate_infer_fn(batch, SOS_token=trg_sos_idx, EOS_token=trg_eos_idx, PAD_token=trg_pad_idx):

    tensors, targets, t_source = [], [], []

    # Gather in lists, and encode labels as indices
    # for waveform, _, label, *_, ut_id in batch:
    for waveform, smp_freq, label, spk_id, ut_id, *_ in batch:
        # label=re.sub(r"[#^$?:;'.!]+|<unk>","",label)

        label = re.sub(r"[#^$,?:;.!]+|<unk>", "", label)

        if "ignore_time_segment_in_scoring" in label:
            continue
        spec = spec_transform(waveform)  # .to(device)
        spec = melspec_transform(spec).to(device)
        npads = 1000
        # if spec.size(2)>1000:
        #    npads = 500
        # spec = F.pad(spec, (0,npads), mode='constant',value=0)
        t_source += [spec.size(2)]

        tensors += spec

        del spec
        if bpe_flag == True:
            # tg=torch.LongTensor([sp.bos_id()] + sp.encode_as_ids(label.upper()) + [sp.eos_id()])
            tg = torch.LongTensor(
                [sp.bos_id()] + sp.encode_as_ids(label) + [sp.eos_id()])
        else:
            tg = torch.LongTensor(
                text_transform.text_to_int("^"+label.lower()+"$"))

        targets += [tg.unsqueeze(0)]
        del waveform
        del label
    if tensors:
        tensors = pad_sequence(tensors, 0)
        targets = pad_sequence(targets, PAD_token)
        len_out = torch.full((len(t_source),), tensors.size(2))
        return tensors.squeeze(1), targets.squeeze(1), len_out
    else:
        return None


def evaluate(model):
    batch_size = 1
    file_dict = 'librispeech.lex'
    words = load_dict(file_dict)

    path = os.getcwd()+'/trained_model/bpe_seq2seq_small_256/'

    model = avg_models(model, path, 30, 100)
    model.eval()
    w_ctc = float(sys.argv[1])

    set_ = 'test'
    beam_size = 10
    m = 5 / 200  # for deciding maximun length
    # p = 33 #for deciding maximun length for 5000
    p = 30  # for deciding maximun length for 256
    for set_ in "test-clean", "test-other":  # "dev-clean", "dev-other":
        print(set_)

        test_dataset = torchaudio.datasets.LIBRISPEECH(
            "/falavi/corpora", url=set_, download=False)
        data_loader = torch.utils.data.DataLoader(
            test_dataset, pin_memory=False, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_infer_fn)

        for batch in data_loader:
            # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            trg_expect = batch[1][:, 1:].to(device)
            # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
            trg = batch[1][:, :-1].to(device)
            for trg_expect_ in trg_expect:
                if bpe_flag == True:
                    print(set_, "EXPECTED:", sp.decode(
                        trg_expect_.squeeze(0).tolist()).lower())
                else:
                    print(set_, "EXPECTED:", re.sub(
                        r"[#^$]+", "", text_transform.int_to_text(trg_expect_.squeeze(0))))
            valid_length = batch[2]

            for spec_, v_l in zip(batch[0], valid_length):
                if spec_.size(1) < 200:
                    max_len = int(p - spec_.size(1) * m)
                else:
                    # max_len=int(spec_.size(1) / 20) #for 5000
                    max_len = int(spec_.size(1) / 12)  # for 256
                min_len = int(max_len * 0.6)
                print("MAX-MIN:", max_len, min_len,
                      spec_.size(1), trg_expect_.size())
                for n in range(1, n_enc_exits+1):
                    encoder_output = model._encoder_(
                        spec_.unsqueeze(0), v_l.unsqueeze(0), n).to(device)
                    # out,scores,best_combined=beam_search(model,encoder_output, n, spec_.unsqueeze(0), trg, valid_length, max_length=max_utterance_length, beam_size=beam_size, return_best_beam=True, weight_ctc=w_ctc)
                    out, scores, best_combined = beam_search(model, encoder_output, n, spec_.unsqueeze(
                        0), trg, valid_length, max_length=max_len, beam_size=beam_size, return_best_beam=True, weight_ctc=w_ctc)
                    del encoder_output
                    if bpe_flag == True:
                        print(set_, " BEAM_OUT_", n, ":",  apply_lex(
                            sp.decode(best_combined).lower(), words))
                del spec_
                del v_l
            del batch


if __name__ == '__main__':
    evaluate(model)
