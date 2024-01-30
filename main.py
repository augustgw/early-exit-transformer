import torch
import sentencepiece as spm

import train_ae
import train_ctc
from util.conf import get_args

# Parse config from command line arguments

args = get_args()

if args.decoder_mode == 'aed':
    train_ae.main(args)
elif args.decoder_mode == 'ctc':
    train_ctc.main(args)