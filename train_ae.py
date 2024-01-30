import os
import time
import torch
from torch import nn
from torch.optim import AdamW

from util.conf import get_args
from data import get_data_loader
from models.model.early_exit import full_conformer
from util.noam_opt import NoamOpt
from util.model_utils import count_parameters, initialize_weights


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

            valid_lengths = batch_3
            att_dec, encoder = model(src, valid_lengths, trg)

            ctc_target_len = batch_2
            loss_ctc = 0
            loss_ce = 0

            if i % 500 == 0:
                print("EXPECTED:", args.sp.decode(trg_expect[0].tolist()).lower())

            ctc_input_len = torch.full(
                size=(encoder.size(1),), fill_value=encoder.size(2), dtype=torch.long)

            for dec, enc in zip(att_dec, encoder):
                loss_ctc += ctc_loss(enc.permute(1, 0, 2), batch_1,
                                     ctc_input_len, ctc_target_len).to(args.device)
                loss_ce += loss_fn(dec.permute(0, 2, 1), trg_expect)

            del encoder

            loss = 0.3 * loss_ctc + 0.7 * loss_ce

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            epoch_loss += loss.item()

        print('step :', round((i / len_iterator) * 100, 2), '% , loss :',
              loss.item(), 'loss_ce :', loss_ce.item(), 'loss_ctc :', loss_ctc.item())

    return epoch_loss / len_iterator


def run(args, model, total_epoch, best_loss, data_loader, optimizer, loss_fn, ctc_loss):

    train_losses, test_losses, bleus = [], [], []
    prev_loss = 9999999
    nepoch = -1

    moddir = os.getcwd()+'/trained_model/bpe_seq2seq_small_256/'
    os.makedirs(moddir, exist_ok=True)
    initialize_model = False
    best_model = moddir+'{}mod{:03d}-transformer'.format('', nepoch)

    best_lr = moddir+'{}lr{:03d}-transformer'.format('', nepoch)

    if os.path.exists(best_model):
        initialize_model = False
        print('loading model checkpoint:', best_model)
        model.load_state_dict(torch.load(best_model, map_location=args.device))

    if os.path.exists(best_lr):
        print('loading learning rate checkpoint:', best_lr)
        optimizer.load_state_dict(torch.load(best_lr))

    # if initialize_model == True:
    #     total_loss = 0
    #     for step in range(0, 30):
    #         print("Initializing step:", step)
    #         total_loss += train(data_loader_1)
    #         print("TOTAL_LOSS-", step, ":=", total_loss)

    for step in range(nepoch + 1, total_epoch):
        start_time = time.time()

        total_loss = train(args=args, model=model, 
                           iterator=data_loader, optimizer=optimizer, 
                           loss_fn=loss_fn, ctc_loss=ctc_loss)
        print("TOTAL_LOSS-", step, ":=", total_loss)

        thr_l = (prev_loss - total_loss) / total_loss
        if total_loss < prev_loss:
            prev_loss = total_loss
            best_model = moddir+'mod{:03d}-transformer'.format(step)

            print("saving:", best_model)
            torch.save(model.state_dict(), best_model)
            lrate = moddir+'lr{:03d}-transformer'.format(step)
            print("saving:", lrate)
            torch.save(optimizer.state_dict(), lrate)
        else:
            worst_model = moddir+'mod{:03d}-transformer'.format(step)
            print("WORST: not saving:", worst_model)

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


def main():
    # Parse config from command line arguments

    args = get_args()

    torch.multiprocessing.set_start_method('spawn')

    # Load data

    data_loader = get_data_loader(args=args)

    # Define model

    model = full_conformer(trg_pad_idx=args.trg_pad_idx,
                            n_enc_exits=args.n_enc_exits,
                            d_model=args.d_model,
                            enc_voc_size=args.enc_voc_size,
                            dec_voc_size=args.dec_voc_size,
                            max_len=args.max_len,
                            d_feed_forward=args.d_feed_forward,
                            n_head=args.n_heads,
                            n_enc_layers=args.n_enc_layers,
                            n_dec_layers=args.n_dec_layers,
                            features_length=args.n_mels,
                            drop_prob=args.drop_prob,
                            depthwise_kernel_size=args.depthwise_kernel_size,
                            device=args.device).to(args.device)
    
    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(args.num_threads)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    warmup = len(data_loader) * args.n_batch_split
    print("batch_size:", args.batch_size, " num_heads:", args.n_heads, " num_encoder_layers:", args.n_enc_layers, " optimizer:",
            "NOAM[warmup ", warmup, "]", "vocab_size:", args.dec_voc_size, "SOS,EOS,PAD", args.trg_sos_idx, args.trg_eos_idx, args.trg_pad_idx, "data_loader_len:", len(data_loader), "DEVICE:", args.device)

    model.apply(initialize_weights)

    loss_fn = nn.CrossEntropyLoss()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = NoamOpt(args.d_model, warmup, AdamW(params=model.parameters(
    ), lr=0, betas=(0.9, 0.98), eps=args.adam_eps, weight_decay=args.weight_decay))

    run(args=args, model=model, total_epoch=args.epoch, best_loss=args.inf, 
        data_loader=data_loader, optimizer=optimizer, loss_fn=loss_fn, ctc_loss=ctc_loss)


if __name__ == "__main__":
    main()