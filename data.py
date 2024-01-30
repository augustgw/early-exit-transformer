import torch
import torchaudio

from util.data_loader import CollatePaddingFn


def get_data_loader(args):

    # train_dataset1 = torchaudio.datasets.LIBRISPEECH(
    #     "", url="train-clean-100", download=False)
    # train_dataset2 = torchaudio.datasets.LIBRISPEECH(
    #     "", url="train-clean-360", download=False)
    # train_dataset3 = torchaudio.datasets.LIBRISPEECH(
    #     "", url="train-other-500", download=False)
    # train_dataset = torch.utils.data.ConcatDataset(
    #     [train_dataset1, train_dataset2, train_dataset3])

    train_dataset = torchaudio.datasets.LIBRISPEECH(
        "", url="train-clean-100", download=False)

    collate_padding_fn = CollatePaddingFn(args=args)
    data_loader = torch.utils.data.DataLoader(train_dataset, 
                                              pin_memory=False, 
                                              batch_size=args.batch_size,
                                              shuffle=args.shuffle, 
                                              collate_fn=collate_padding_fn, 
                                              num_workers=args.num_workers)
    # data_loader_initial = torch.utils.data.DataLoader(
    # train_dataset1, pin_memory=False, batch_size=args.batch_size, shuffle=args.shuffle, collate_fn=collate_padding_fn, num_workers=args.num_workers)

    return data_loader
