"""
@author : Falavigna
@when : 2023-02-24
@homepage : https://github.com/
"""
from conf import *
import torchaudio
from util.data_loader import collate_fn
import torch.utils.data as data_utils

train_dataset1 = torchaudio.datasets.LIBRISPEECH("/workspace", url="train-clean-100", download=False)
train_dataset2 = torchaudio.datasets.LIBRISPEECH("/workspace", url="train-clean-360", download=False)
# train_dataset3 = torchaudio.datasets.LIBRISPEECH("/workspace", url="train-other-500", download=False)
# train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2,train_dataset3]) 

train_dataset = train_dataset2

# train_dataset = data_utils.Subset(train_dataset2, torch.arange(0,50000))

data_loader = torch.utils.data.DataLoader(train_dataset, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
data_loader_initial = torch.utils.data.DataLoader(train_dataset1, pin_memory=False, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

