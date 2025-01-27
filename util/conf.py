import argparse
import torch
import sentencepiece as spm

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture

    parser.add_argument(
        "--decoder_mode",
        type=str.lower,
        required=True,
        choices=["ctc","aed"],
        default=None,
        help="""
            Required: Whether to use a connectionist temporal 
            classification-based ('ctc') or attention 
            encoder-decoder-based ('aed') decoder.
        """
    )
    
    parser.add_argument(
        "--model_type",
        type=str.lower,
        choices=["early_conformer","early_zipformer", "splitformer"],
        default="early_conformer",
        help="""
            Required: If you use a connectionist temporal 
            classification-based ('ctc') decoder, choose 
            the model you want to use.
        """
    )

    parser.add_argument(
        "--bpe",
        type=bool,
        default=True,
        help="""
            Whether to use BPE-based tokenization with SentencePiece.
            NOTE: Presently, SentencePiece is the only option.
            Therefore, --bpe should be left as True.
        """
    )

    parser.add_argument(
        "--distill",
        type=bool,
        default=False,
        help="""
            Whether to use knowledge distillation.
            NOTE: Presently, knowledge distillation is not implemented.
            Therefore, --distill should be left as False.
        """
    )

    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="/trained_model",
        help="""
            Path to directory in which to save the trained model 
            state_dict and learning rate objects. By default, 
            the model is saved after achieving a new best 
            loss during training.
        """
    )

    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="""
            Path to pre-trained model. In train.py, loads the model
            as a checkpoint from which to begin training. In 
            inference.py, loads the model for inference.
        """
    )

    parser.add_argument(
        "--load_model_dir",
        type=str,
        default=None,
        help="""
            Path to directory containing pre-trained model(s). 
            Must be used in conjunction with --avg_model_start 
            and --avg_model_end. Loads and averages the models 
            from epochs --avg_model_start to --avg_model_end.
            In train.py, the averaged model is used as a
            checkpoint from which to begin training. 
            In inference.py, the averaged model is used
            for inference.
        """
    )

    parser.add_argument(
        "--avg_model_start",
        type=int,
        default=None,
        help="""
            Epoch from which to begin model averaging. 
            Must be used in conjunction with --load_model_dir 
            and --avg_model_end. Loads and averages the models 
            from epochs --avg_model_start to --avg_model_end.
            In train.py, the averaged model is used as a
            checkpoint from which to begin training. 
            In inference.py, the averaged model is used
            for inference.
        """
    )

    parser.add_argument(
        "--avg_model_end",
        type=int,
        default=None,
        help="""
            Epoch from which to end model averaging. 
            Must be used in conjunction with --load_model_dir 
            and --avg_model_start. Loads and averages the models 
            from epochs --avg_model_start to --avg_model_end.
            In train.py, the averaged model is used as a
            checkpoint from which to begin training. 
            In inference.py, the averaged model is used
            for inference.
        """
    )

    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="""
            Shuffles training data upon loading.
        """
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10000,
        help="""
            Number of training epochs.
        """
    )

    # GPU device settings

    parser.add_argument(
        "--n_threads",
        type=int,
        default=10,
        help="""
            Sets number of threads for intraop parallelism on CPU.
            See PyTorch torch.set_num_threads method.
        """
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        default=10,
        help="""
            Sets number of GPU workers for loading data.
        """
    )

    # Model parameter settings

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="""
            Batch size during training and inference.
        """
    )

    parser.add_argument(
        "--n_batch_split",
        type=int,
        default=4,
        help="""
            In each batch, items are ordered by length 
            and split into this number of sub-batches, 
            in order to minimize padding and maximize 
            GPU performance.
        """
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=2000,
        help="""
            Maximum length in terms of number of characters for model inputs.
        """
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="""
            Dimensionality of the model.
        """
    )

    parser.add_argument(
        "--n_enc_layers_per_exit",
        type=int,
        default=2,
        help="""
            Number of encoder layers per exit (where number of
            exits is determined by --n_enc_exits). 
            For example, --n_enc_layers_per_exit=2 and --n_enc_exits=6
            results in a encoder with 6 exits and 12 total layers,
            with an exit occurring every 2 layers.
        """
    )

    parser.add_argument(
        "--n_enc_exits",
        type=int,
        default=6,
        help="""
            Number of exits in the model (where number of
            layers per exit is determined by --n_enc_layers_per_exit). 
            For example, --n_enc_layers_per_exit=2 and --n_enc_exits=6
            results in a encoder with 6 exits and 12 total layers,
            with an exit occurring every 2 layers.
        """
    )

    parser.add_argument(
        "--n_dec_layers",
        type=int,
        default=6,
        help="""
            Number of decoder layers in each exit in the encoder.
        """
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="""
            Number of attention heads in each encoder layer.
        """
    )

    parser.add_argument(
        "--d_feed_forward",
        type=int,
        default=2048,
        help="""
            Dimensionality of the feed-forward network.
        """
    )

    parser.add_argument(
        "--aed_ce_weight",
        type=int,
        default=0.7,
        help="""
            For AED models: weight coefficient for the 
            cross-entropy loss.
        """
    )

    parser.add_argument(
        "--aed_ctc_weight",
        type=int,
        default=0.3,
        help="""
            For AED models: weight coefficient for the
            CTC loss.
        """
    )

    parser.add_argument(
        "--drop_prob",
        type=int,
        default=0.1,
        help="""
            Probability of a given element of the input to be 
            randomly dropped during training.
        """
    )

    parser.add_argument(
        "--depthwise_kernel_size",
        type=int,
        default=31,
        help="""
            Kernel size of the depthwise convolutions in each Conformer block.
        """
    )

    parser.add_argument(
        "--max_utterance_length",
        type=int,
        default=360,
        help="""
            Input items longer than this value in terms of number of labels
            will be dropped during training.
        """
    )

    parser.add_argument(
        "--lexicon_path",
        type=str,
        default="lexicon.txt",
        help="""
            Path to lexicon file.
            NOTE: The current implementation overwrites this value and
            uses a custom lexicon file for SentencePiece tokenization.
        """
    )

    parser.add_argument(
        "--tokens_path",
        type=str,
        default="tokens.txt",
        help="""
            Path to tokens file.
            NOTE: The current implementation overwrites this value and
            uses a custom tokens file for SentencePiece tokenization.
        """
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="""
            Sample rate used in preprocessing raw audio inputs to the model.
        """
    )

    parser.add_argument(
        "--n_fft",
        type=int,
        default=512,
        help="""
            Size of Fast Fourier Transform used to generate spectrogram of
            raw audio input during preprocessing.
        """
    )

    parser.add_argument(
        "--win_length",
        type=int,
        default=320,
        help="""
            Window length used to generate spectrogram of
            raw audio input during preprocessing.
        """
    )

    parser.add_argument(
        "--hop_length",
        type=int,
        default=160,
        help="""
            Length of hop between STFT windows used to generate spectrogram
            of raw audio input during preprocessing.
        """
    )

    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="""
            Number of mel filterbanks used to compute STFT
            of raw audio input during preprocessing.
        """
    )

    # Optimizer parameter settings

    parser.add_argument(
        "--init_lr",
        type=int,
        default=1e-5,
        help="""
            Initial learning rate during training.
        """
    )

    parser.add_argument(
        "--adam_eps",
        type=int,
        default=1e-9,
        help="""
            Epsilon parameter used in AdamW optimization algorithm.
        """
    )

    parser.add_argument(
        "--weight_decay",
        type=int,
        default=5e-4,
        help="""
            Weight decay coefficient used in AdamW optimization algorithm.
        """
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=-1,
        help="""
            Number of learning rate warmup steps.
            Default behavior (= -1): Warmup for the length of the
            dataloader.
        """
    )

    parser.add_argument(
        "--clip",
        type=int,
        default=1.0,
        help="""
            Gradient norms higher than this value will be clipped during training.
            See PyTorch torch.nn.utils.clip_grad_norm_ function.
        """
    )

    # Inference settings

    parser.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="""
            Beam size for AED inference.
        """
    )

    parser.add_argument(
        "--pen_alpha",
        type=int,
        default=1.0,
        help="""
            Sequence length penalty alpha for AED inference.
        """
    )

    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()

    conf = vars(args)

    conf["decoder_mode"] = args.decoder_mode.lower()

    conf["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf["src_pad_idx"] = 0
    conf["trg_pad_idx"] = 30
    conf["trg_sos_idx"] = 1
    conf["trg_eos_idx"] = 31
    conf["enc_voc_size"] = 32
    conf["dec_voc_size"] = 32

    if args.bpe == True:
        conf["sp"] = spm.SentencePieceProcessor()
        conf["sp"].load('sentencepiece/build/libri.bpe-256.model')
        conf["src_pad_idx"] = 0
        conf["trg_pad_idx"] = 126
        conf["trg_sos_idx"] = 1
        conf["trg_eos_idx"] = 2
        conf["enc_voc_size"] = conf["sp"].get_piece_size()
        conf["dec_voc_size"] = conf["sp"].get_piece_size()
        conf["lexicon"] = "sentencepiece/build/librispeech-bpe-256.lex"
        conf["tokens"] = "sentencepiece/build/librispeech-bpe-256.tok"
 
    conf["inf"] = float('inf')

    return args