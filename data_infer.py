"""
@author : Falavigna
@when : 2023-02-24
@homepage : https://github.com/
"""
from conf_infer import *
import torchaudio
from util.data_loader import collate_infer_fn
'''
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(ext=('.en', '.de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                     batch_size=batch_size,
                                                     device=device)

src_pad_idx = loader.source.vocab.stoi['<pad>']
trg_pad_idx = loader.target.vocab.stoi['<pad>']
trg_sos_idx = loader.target.vocab.stoi['<sos>']

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
'''

test_clean = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="test-clean", download=False)
dev_clean = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="dev-clean", download=False)
test_other = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="test-other", download=False)
dev_other = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url="dev-other", download=False)

data_loader = torch.utils.data.DataLoader(dataset=test_clean, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_infer_fn, num_workers=num_workers)

