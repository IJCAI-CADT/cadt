import sys
import torch
import torch.nn as nn
from utils.data_argumentation import data_argumentation


class CNNEncoder(nn.Module):
    def __init__(self, t_size):
        super(CNNEncoder, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.feature_10 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=128, kernel_size=(7, ), padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(5,), padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.feature_11 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(5, ), padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=(3, ), padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=128),
        )

        self.feature_20 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=128, kernel_size=(7,), padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(21,), stride=1, padding=10),
        )

        self.feature_22 = nn.AvgPool1d(kernel_size=128)
        self.feature_21 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=(5,), padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=126)
        )
        self.linear1 = nn.Sequential(nn.Linear(128, 128))
        self.linear2 = nn.Sequential(nn.Linear(128, 128))

        self.fc11 = nn.Sequential(nn.Linear(128, t_size))
        self.fc21 = nn.Sequential(nn.Linear(128, t_size))

    def forward(self, z, with_argumentation=False, with_connection=True):
        data = z.cuda()
        data = data.transpose(1, 2)
        # print(data.shape)
        if with_argumentation:
            data = data_argumentation(data, series_length=128, sub_seq_length=32)
            data = data_argumentation(data, series_length=128, sub_seq_length=32)
        c = self.feature_10(data)
        d = self.feature_20(data)
        e = self.feature_22(d)

        if with_connection:
            d2 = d + torch.transpose(self.linear1(torch.transpose(c, 1, 2)), 1, 2)
            c2 = c + torch.transpose(self.linear2(torch.transpose(d, 1, 2)), 1, 2)
            # d2 = torch.concat([d, torch.transpose(self.linear1(torch.transpose(c, 1, 2)), 1, 2)], dim=1)
            # c2 = torch.concat([c, torch.transpose(self.linear2(torch.transpose(d, 1, 2)), 1, 2)], dim=1)
        else:
            d2 = d
            c2 = c

        data1 = self.feature_11(c2)
        data1 = data1.flatten(1, 2)
        f1 = self.fc11(data1)

        data2 = self.feature_21(d2)
        data2 = data2 + e
        data2 = data2.flatten(1, 2)
        f2 = self.fc21(data2)

        return f1, f2


class Classifier(nn.Module):
    def __init__(self, input_size, y_dim):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_size, y_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        r = self.linear(inputs)
        return r


class Discriminator(nn.Module):
    def __init__(self, output_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(output_size), 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        feature = inputs.view(inputs.size(0), -1)
        validity = self.model(feature)
        return validity