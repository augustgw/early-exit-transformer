"""
@author : Falavigna
@when : 2023-02-24
@homepage : https://github.com/
"""
from conf import *
import torchaudio
from util.data_loader import collate_fn
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

train_dataset1 = torchaudio.datasets.LIBRISPEECH("/workspace", url="train-clean-100", download=False)
# train_dataset2 = torchaudio.datasets.LIBRISPEECH("/workspace", url="train-clean-360", download=False)
# train_dataset3 = torchaudio.datasets.LIBRISPEECH("/workspace", url="train-other-500", download=False)
# train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2,train_dataset3]) 
train_dataset = train_dataset1

data_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
data_loader_initial = torch.utils.data.DataLoader(train_dataset1, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

