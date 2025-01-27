import os
import torch
from torch import nn, optim
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from util.conf import get_args
from data import get_data_loader
from models.model.early_exit import Early_conformer, full_conformer, Early_zipformer, Splitformer
from util.beam_infer import BeamInference
from util.noam_opt import NoamOpt
from util.model_utils import count_parameters, initialize_weights, avg_models


def train(args, model, iterator, optimizer, loss_fn, ctc_loss):

    model.train()
    epoch_loss = 0
    len_iterator = len(iterator)
    print(len_iterator)

    for i, c_batch in enumerate(iterator):
        if len(c_batch) != args.n_batch_split:
            continue

        for batch_0, batch_1, batch_2, batch_3 in c_batch:

            src = batch_0.to(args.device)
            # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
            trg = batch_1[:, :-1].to(args.device)
            # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            trg_expect = batch_1[:, 1:].to(args.device)
            ctc_target_len = batch_2
            valid_lengths = batch_3

            if args.decoder_mode == 'aed':
                att_dec, encoder = model(src, valid_lengths, trg)
                loss_ctc = 0
                loss_ce = 0

                ctc_input_len = torch.full(
                    size=(encoder.size(1),), fill_value=encoder.size(2), dtype=torch.long)

                for dec, enc in zip(att_dec, encoder):
                    loss_ctc += ctc_loss(enc.permute(1, 0, 2), batch_1,
                                         ctc_input_len, ctc_target_len).to(args.device)
                    loss_ce += loss_fn(dec.permute(0, 2, 1), trg_expect)

                del encoder

                loss = args.aed_ce_weight * loss_ce + args.aed_ctc_weight * loss_ctc

            elif args.decoder_mode == 'ctc':
                encoder = model(src, valid_lengths)
                loss_ctc = 0

                ctc_input_len = torch.full(
                    size=(encoder.size(1),), fill_value=encoder.size(2), dtype=torch.long)

                for enc in encoder:
                    loss_ctc += ctc_loss(enc.permute(1, 0, 2), batch_1,
                                         ctc_input_len, ctc_target_len).to(args.device)
                del encoder

                loss = loss_ctc

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item()

            if i % 500 == 0:
                inf = BeamInference(args)
                print("EXPECTED:", args.sp.decode(
                    trg_expect[0].tolist()).lower())
                best_combined = inf.ctc_cuda_predict(
                    emission=enc[0].unsqueeze(0), tokens=args.tokens)
                print("CTC_OUT at [", i, "]:", args.sp.decode(
                    best_combined[0][0].tokens).lower())

        if args.decoder_mode == 'aed':
            print('step: ', round((i / len_iterator) * 100, 2), '% , loss_total: ',
                  loss.item(), 'loss_ce: ', loss_ce.item(), 'loss_ctc: ', loss_ctc.item())
        elif args.decoder_mode == 'ctc':
            print('step: ', round((i / len_iterator) * 100, 2), '% , loss_total: ',
                  loss.item(), 'loss_ctc: ', loss_ctc.item())

    loss_total = epoch_loss / len_iterator

    return loss_total


def run(args, model, total_epoch, best_loss, data_loader, optimizer, loss_fn, ctc_loss):
    loss_prev = 9999999
    nepoch = -1

    moddir = os.getcwd() + '/' + args.save_model_dir + '/'
    os.makedirs(moddir, exist_ok=False)

    best_model = moddir+'{}mod{:03d}-transformer'.format('', nepoch)
    best_lr = moddir+'{}lr{:03d}-transformer'.format('', nepoch)

    if os.path.exists(best_model):
        print('loading model checkpoint:', best_model)
        model.load_state_dict(torch.load(best_model, map_location=args.device))

    if os.path.exists(best_lr):
        print('loading learning rate checkpoint:', best_lr)
        optimizer.load_state_dict(torch.load(best_lr))

    for step in range(nepoch + 1, total_epoch):
        loss_total = train(args=args, model=model,
                           iterator=data_loader, optimizer=optimizer,
                           loss_fn=loss_fn, ctc_loss=ctc_loss)
        writer.add_scalar("Total loss", loss_total, step)
        print("LOSS_TOTAL-", step, ":=", loss_total)

        if loss_total < loss_prev:
            loss_prev = loss_total
            best_model = moddir + 'mod{:03d}-transformer'.format(step)

            print("saving:", best_model)
            torch.save(model.state_dict(), best_model)
            lrate = moddir + 'lr{:03d}-transformer'.format(step)
            print("saving:", lrate)
            torch.save(optimizer.state_dict(), lrate)

        else:
            worst_model = moddir + 'mod{:03d}-transformer'.format(step)
            print("WORST: not saving:", worst_model)


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
            
        if args.model_type == 'early_zipformer':
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
            
        elif args.model_type == 'splitformer':
            model = Splitformer(src_pad_idx=args.src_pad_idx,
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

    # Otherwise, initialize weights with Xavier uniform distribution.
    else:
        model.apply(initialize_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.n_threads)

    #
    #   DATALOADER
    #

    # Load data
    data_loader = get_data_loader(args=args)

    # If default, set warmup to length of dataloader
    if args.warmup == -1:
        args.warmup = len(data_loader) * args.n_batch_split

    #
    #   RUN TRAINING
    #
    
    print("batch_size:", args.batch_size, " num_heads:", args.n_heads, " num_encoder_layers:", args.n_enc_layers_per_exit, " optimizer:",
          "NOAM[warmup ", args.warmup, "]", "vocab_size:", args.dec_voc_size, "SOS,EOS,PAD", args.trg_sos_idx, args.trg_eos_idx, args.trg_pad_idx, "data_loader_len:", len(data_loader), "DEVICE:", args.device)

    

    # Define loss functions (Note: In ctc decoder mode, loss_fn is not used)
    loss_fn = nn.CrossEntropyLoss()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = NoamOpt(args.d_model, args.warmup, AdamW(params=model.parameters(
    ), lr=0, betas=(0.9, 0.98), eps=args.adam_eps, weight_decay=args.weight_decay))

    run(args=args, model=model, total_epoch=args.n_epochs, best_loss=args.inf,
        data_loader=data_loader, optimizer=optimizer, loss_fn=loss_fn, ctc_loss=ctc_loss)


if __name__ == "__main__":
    # Initialize the SummaryWriter for TensorBoard
    # Its output will be written to ./runs/
    writer = SummaryWriter()

    main()
