import os
import torch
from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def avg_models(args, model, path, start, end):
    if start > end:
        raise ValueError("--avg_model_start must be less than --avg_model_end")

    n_epoch = start

    best_model = os.getcwd() + '/' + path + '/' + \
        'mod{:03d}-transformer'.format(n_epoch)
    model.load_state_dict(torch.load(
        best_model, map_location=args.device))
    m1 = model.state_dict()
    nc = 1

    for n_epoch in range(n_epoch+1, end+1):
        best_model = path + '/mod{:03d}-transformer'.format(n_epoch)

        if os.path.exists(best_model):
            print("Averaging with:", best_model)
            model.load_state_dict(torch.load(
                best_model, map_location=torch.device(args.device)))

            m2 = model.state_dict()
            for key in m2:
                m1[key] = m2[key] + m1[key]

            nc = nc + 1
            del m2

    for key in m1:
        m1[key] = m1[key] / nc

    model.load_state_dict(m1)
    del m1
    return model
