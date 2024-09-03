import os
import sys
import re
from torch import nn, optim
import torchaudio
from torchaudio.models.decoder import ctc_decoder

from data import get_infer_data_loader
from models.model.early_exit import Early_conformer, full_conformer, Early_zipformer, Early_conformer_plus
from util.beam_infer import BeamInference
from util.conf import get_args
from util.data_loader import text_transform
from util.epoch_timer import epoch_time
from util.model_utils import *
from util.tokenizer import *


def evaluate_batch_ae(args, model, batch, valid_len, split, inf, vocab):
    beam_size = 10
    m = 5 / 200  # for deciding maximum length
    # p = 33 # for deciding maximum length for 5000
    p = 30  # for deciding maximum length for 256

    # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
    trg_expect = batch[1][:, 1:].to(args.device)
    
    for spec_, v_l, trg_expect_ in zip(batch[0], valid_len, trg_expect):
        
        if args.bpe == True:
            print(split, "\nEXPECTED:", args.sp.decode(
                trg_expect_.squeeze(0).tolist()).lower())
        else:
            print(split, "\nEXPECTED:", re.sub(
                r"[#^$]+", "", text_transform.int_to_text(trg_expect_.squeeze(0))))

        if spec_.size(1) < 200:
            max_len = int(p - spec_.size(1) * m)
        else:
            max_len = int(spec_.size(1) / 12)  # for 256
        min_len = int(max_len * 0.6)
        # print("MAX-MIN:", max_len, min_len,
        #       spec_.size(1), trg_expect_.size())

        for n in range(1, args.n_enc_exits+1):
            encoder_output = model._encoder_(
                spec_.unsqueeze(0), v_l.unsqueeze(0), n).to(args.device)

            out, scores, best_combined = inf.beam_search(
                model, encoder_output=encoder_output, layer_n=n,
                max_length=max_len, beam_size=beam_size,
                return_best_beam=True)

            del encoder_output

            if args.bpe == True:
                print(split, " BEAM_OUT_", n, ":",  apply_lex(
                    args.sp.decode(best_combined).lower(), vocab))

        del spec_
        del v_l

    return


def evaluate_batch_ctc(args, model, batch, valid_len, split, inf, vocab):
    encoder = model(batch[0].to(args.device), valid_len)
    i = 0

    for enc in encoder:
        i = i+1

        best_combined = inf.ctc_cuda_predict(enc, args.tokens)
        
        for best_ in best_combined:
            if args.bpe == True:
                print(split, "BEAM_OUT_", i, ":", apply_lex(
                    args.sp.decode(best_[0].tokens).lower(), vocab))
            else:
                print(split, "BEAM_OUT_", i, ":",  apply_lex(
                    re.sub(r"[#^$]+", "", best_.lower()), vocab))

    return


def run(args, model, data_loader, split, inf, vocab):
    for batch in data_loader:
        # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
        trg_expect = batch[1][:, 1:].to(args.device)
        # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
        # trg = batch[1][:, :-1].to(args.device)

        for trg_expect_ in trg_expect:
            if args.bpe == True:
                print(split, "EXPECTED:", args.sp.decode(
                    trg_expect_.squeeze(0).tolist()).lower())
            else:
                print(split, "EXPECTED:", re.sub(
                    r"[#^$]+", "", text_transform.int_to_text(trg_expect_.squeeze(0))))

        valid_len = batch[2]

        if args.decoder_mode == 'aed':
            evaluate_batch_ae(args, model, batch,
                              valid_len, split, inf, vocab)
        elif args.decoder_mode == 'ctc':
            evaluate_batch_ctc(args, model, batch,
                               valid_len, split, inf, vocab)

    return


def main():
    #
    #   CONFIG
    #

    # Parse config from command line arguments
    args = get_args()

    #
    #   MODEL
    #

    # Define model
    if args.decoder_mode == 'aed':
        model = full_conformer(trg_pad_idx=args.trg_pad_idx,
                               n_enc_exits=args.n_enc_exits,
                               d_model=args.d_model,
                               enc_voc_size=args.enc_voc_size,
                               dec_voc_size=args.dec_voc_size,
                               max_len=args.max_len,
                               d_feed_forward=args.d_feed_forward,
                               n_head=args.n_heads,
                               n_enc_layers=args.n_enc_layers_per_exit,
                               n_dec_layers=args.n_dec_layers,
                               features_length=args.n_mels,
                               drop_prob=args.drop_prob,
                               depthwise_kernel_size=args.depthwise_kernel_size,
                               device=args.device).to(args.device)

    elif args.decoder_mode == 'ctc':
        if args.model_type == 'early_conformer':
            model = Early_conformer(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=args.n_mels,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)
            
        elif args.model_type == 'early_zipformer':
            model = Early_zipformer(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=args.n_mels,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)
            
        elif args.model_type == 'early_conformer_plus':
            model = Early_conformer_plus(src_pad_idx=args.src_pad_idx,
                                    n_enc_exits=args.n_enc_exits,
                                    d_model=args.d_model,
                                    enc_voc_size=args.enc_voc_size,
                                    dec_voc_size=args.dec_voc_size,
                                    max_len=args.max_len,
                                    d_feed_forward=args.d_feed_forward,
                                    n_head=args.n_heads,
                                    n_enc_layers=args.n_enc_layers_per_exit,
                                    features_length=args.n_mels,
                                    drop_prob=args.drop_prob,
                                    depthwise_kernel_size=args.depthwise_kernel_size,
                                    device=args.device).to(args.device)

    else:
        raise ValueError(
            "Invalid decoder mode. Use either \"aed\" or \"ctc\" with --decoder_mode.")

    # If model checkpoint path is provided, load it.
    # (Overrides --load_model-dir)
    if args.load_model_path != None:
        path = os.getcwd() + '/' + args.load_model_path
        model.load_state_dict(torch.load(
            path, map_location=args.device))

    # If model checkpoint dir is provided, check that
    # the epochs to begin and end averaging are also
    # provided. If so, average the specified models.
    elif None not in (args.load_model_dir, args.avg_model_start, args.avg_model_end):
        model = avg_models(args, model, args.load_model_dir,
                           args.avg_model_start, args.avg_model_end)

    # If neither option has been provided, then raise error.
    else:
        raise ValueError(
            "Invalid model loading config. Use either --load_model_path for a single model or --load_model_dir/--avg_model_start/--avg_model_end for an average of models.")

    model.eval()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    # print("batch_size:", batch_size, " num_heads:", n_heads, " num_encoder_layers:",
    #     n_enc_layers, "vocab_size:", dec_voc_size, "DEVICE:", device)

    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)

    # Used to access various inference functions, see util/beam_infer
    inf = BeamInference(args=args)

    file_dict = 'librispeech.lex'
    vocab = load_dict(file_dict)

    for split in ["test-clean", "test-other"]:  # "dev-clean", "dev-other":
        print(split)

        # Load data split
        data_loader = get_infer_data_loader(
            args=args, split=split, shuffle=False)

        run(model=model, args=args, data_loader=data_loader,
            split=split, inf=inf, vocab=vocab)


if __name__ == '__main__':
    main()
