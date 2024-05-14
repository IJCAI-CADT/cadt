import random
import torch


def segment_exchange(input_x, series_length=10, sub_seq_length=1):
    t = [i for i in range(series_length)]
    i = random.randint(0, series_length - 1 - 2 * sub_seq_length - 1)
    j = random.randint(sub_seq_length + 1, series_length - 1 - sub_seq_length)
    while j - i <= sub_seq_length:
        j = random.randint(sub_seq_length + 1, series_length - 1 - sub_seq_length)
    for v in range(sub_seq_length):
        t[i + v] = j + v
        t[j + v] = i + v
    idx = torch.LongTensor(t).cuda()
    r = input_x.index_select(dim=1, index=idx)
    return r


def segment_remove(input_x, series_length=10, sub_seq_length=1):
    t = [True for _ in range(series_length)]
    i = random.randint(0, series_length - 1 - sub_seq_length - 1)
    for v in range(sub_seq_length):
        t[i + v] = False
    b = torch.masked_select(input_x, torch.tensor(t, device='cuda')).cuda()
    c = torch.zeros(input_x.shape[0], sub_seq_length).cuda()
    b = b.view(input_x.shape[0], -1)
    r = torch.cat([b, c], dim=1)
    return r


def segment_shuffle(input_x, series_length=10, sub_seq_length=2):
    t = [i for i in range(series_length)]
    i = random.randint(0, series_length - 1 - sub_seq_length - 1)
    for v in range(int(sub_seq_length * 0.5)):
        t[i + v] = t[i + sub_seq_length - 1 - v]
    idx = torch.LongTensor(t).cuda()
    r = input_x.index_select(dim=1, index=idx)
    return r


def segment_noise(input_x, var=0.1):
    r = torch.randn_like(input_x, requires_grad=False).cuda() * var + input_x
    return r


def data_argumentation(input_x, series_length=10, sub_seq_length=1):
    input_num = input_x.shape[0]
    r = []
    for i in range(input_num):
        t = random.random()
        if t < 0.1:
            r.append(segment_noise(input_x[i]))
        elif t < 0.4:
            r.append(segment_shuffle(input_x[i], series_length, sub_seq_length))
        elif t < 0.7:
            r.append(segment_exchange(input_x[i], series_length, sub_seq_length))
        else:
            r.append(segment_remove(input_x[i], series_length, sub_seq_length))
    return torch.stack(r).cuda()