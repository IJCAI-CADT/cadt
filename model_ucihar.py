from torch.autograd import Variable
from main import parser
import random
import numpy as np
import torch
import torch.nn as nn
from data_helper import U_data
from torch.utils.data import DataLoader


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


class CATDModel(nn.Module):
    def __init__(self, args):
        super(CATDModel, self).__init__()

        self.args = args
        self.input_size = args.x_dim
        self.hidden_size = args.d_AE
        self.y_dims = args.n_class

        t_size = 8
        self.tsize = t_size

        self.share_encoder = CNNEncoder(t_size)
        self.classifier = Classifier(t_size, self.y_dims)
        self.classifier2 = Classifier(t_size, self.y_dims)

        self.discriminator = Discriminator(t_size + 1)
        self.discriminator2 = Discriminator(t_size)

        lr = 1e-3
        self.share_encoder_optimizer = torch.optim.AdamW(self.share_encoder.parameters(), lr=lr)
        self.classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        self.classifier2_optimizer = torch.optim.AdamW(self.classifier2.parameters(), lr=lr)
        self.discriminator2_optimizer = torch.optim.AdamW(self.discriminator2.parameters(), lr=lr)

        self.zz = nn.Linear(self.tsize, 2)
        self.ae_loss = torch.nn.MSELoss(reduction='mean')
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.a = torch.eye(self.y_dims).cuda()
        self.c = None

    def reset(self):
        self.share_encoder = CNNEncoder(self.tsize).cuda()
        self.classifier = Classifier(self.tsize, self.y_dims).cuda()
        self.classifier2 = Classifier(self.tsize, self.y_dims).cuda()
        self.discriminator = Discriminator(self.tsize + 1).cuda()
        self.discriminator2 = Discriminator(self.tsize).cuda()

        lr = 1e-3
        self.share_encoder_optimizer = torch.optim.AdamW(self.share_encoder.parameters(), lr=lr)
        self.classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)
        self.classifier2_optimizer = torch.optim.AdamW(self.classifier2.parameters(), lr=lr)
        self.discriminator2_optimizer = torch.optim.AdamW(self.discriminator2.parameters(), lr=lr)

    def init_center_c(self, train_loader):
        n_samples = torch.zeros(self.y_dims).cuda()
        c = torch.zeros(self.y_dims, self.tsize).cuda()
        with torch.no_grad():
            for ind, (xs, ys) in enumerate(train_loader):
                xs = torch.squeeze(xs.to(torch.float32), -1)
                xs = xs.cuda()
                ys = torch.as_tensor(ys, dtype=torch.long).cuda()
                ys = ys.cuda()
                y_onehot = self.a[ys]
                zsc, zs = self.share_encoder(xs)
                n_samples = n_samples + torch.sum(y_onehot, dim=0)
                c = c + torch.matmul(y_onehot.T, zsc)
        c = c / torch.unsqueeze(n_samples, 1)
        self.c = c.clone().detach()

    def train_step(self, x_fake, y_fake, x_real, with_argumentation=True, change_center=False, kl_w=0.0,
                   dis_w=1.0):
        zsc, zs = self.share_encoder(x_fake, with_argumentation=False)
        ztc, zt = self.share_encoder(x_real, with_argumentation=False)
        if change_center:
            ce_loss = self.ce_loss(self.classifier(zsc), y_fake)

            real_label = Variable(torch.Tensor(x_fake.size(0), 1).fill_(0), requires_grad=False).cuda()
            fake_label = Variable(torch.Tensor(x_real.size(0), 1).fill_(1), requires_grad=False).cuda()
            domain_class_loss = self.bce_loss(self.discriminator2(zt), fake_label) + \
                                self.bce_loss(self.discriminator2(zs), real_label)

            # print(ce_loss.item(), kl_loss.item(), dloss.item())
            if with_argumentation:
                _, zs2 = self.share_encoder(x_fake, with_argumentation=True)
                _, zt2 = self.share_encoder(x_real, with_argumentation=True)
                domain_class_loss2 = self.bce_loss(self.discriminator2(zt2), fake_label) + \
                                     self.bce_loss(self.discriminator2(zs2), real_label)
            else:
                domain_class_loss2 = 0.0

            total_loss = ce_loss + (domain_class_loss2 + domain_class_loss) * 1.0
            self.share_encoder_optimizer.zero_grad()
            self.classifier_optimizer.zero_grad()
            self.discriminator2_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.share_encoder.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.discriminator2.parameters(), max_norm=0.5)
            self.share_encoder_optimizer.step()
            self.classifier_optimizer.step()
            self.discriminator2_optimizer.step()
            return total_loss.item()

        py = self.c[y_fake].detach()

        dloss = torch.tensor(0.0).cuda()
        zsd = []
        ztd = []
        for i in range(self.y_dims):
            for j in range(i + 1, self.y_dims):
                dloss = dloss + self.ae_loss(self.c[i], self.c[j])
            zsd.append(torch.norm(zsc - self.c[i].detach(), dim=1, keepdim=True))
            ztd.append(torch.norm(ztc - self.c[i].detach(), dim=1, keepdim=True))
        zsd = torch.concat(zsd, dim=1)
        ztd = torch.concat(ztd, dim=1)
        zsd = torch.min(zsd, dim=1, keepdim=True).values / self.tsize
        ztd = torch.min(ztd, dim=1, keepdim=True).values / self.tsize

        kl_loss = self.ae_loss(zsc, py)
        ce_loss = self.ce_loss(self.classifier(zsc), y_fake)

        real_label = Variable(torch.Tensor(x_fake.size(0), 1).fill_(0), requires_grad=False).cuda()
        fake_label = Variable(torch.Tensor(x_real.size(0), 1).fill_(1), requires_grad=False).cuda()
        dis_loss = self.bce_loss(self.discriminator(torch.concat([ztc, ztd], dim=1)), real_label) + \
                   self.bce_loss(self.discriminator(torch.concat([zsc, zsd], dim=1)), fake_label)

        domain_class_loss = self.bce_loss(self.discriminator2(zt), fake_label) + \
                            self.bce_loss(self.discriminator2(zs), real_label)

        # print(ce_loss.item(), kl_loss.item(), dloss.item())
        if with_argumentation:
            _, zs2 = self.share_encoder(x_fake, with_argumentation=True)
            _, zt2 = self.share_encoder(x_real, with_argumentation=True)
            domain_class_loss2 = self.bce_loss(self.discriminator2(zt2), fake_label) + \
                                 self.bce_loss(self.discriminator2(zs2), real_label)
        else:
            domain_class_loss2 = 0.0

        total_loss = ce_loss * 1.0 + kl_loss * kl_w + dis_loss * dis_w + 1.0 * (domain_class_loss + domain_class_loss2)

        self.share_encoder_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        self.discriminator2_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.share_encoder.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_(self.discriminator2.parameters(), max_norm=0.5)
        self.share_encoder_optimizer.step()
        self.classifier_optimizer.step()
        self.discriminator2_optimizer.step()

        real_label = Variable(torch.Tensor(x_fake.size(0), 1).fill_(1), requires_grad=False).cuda()
        fake_label = Variable(torch.Tensor(x_real.size(0), 1).fill_(0), requires_grad=False).cuda()
        dis_loss = self.bce_loss(self.discriminator(torch.concat([ztc.detach(), ztd.detach()], dim=1)), real_label) + \
                   self.bce_loss(self.discriminator(torch.concat([zsc.detach(), zsd.detach()], dim=1)), fake_label)
        self.discriminator_optimizer.zero_grad()
        dis_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=0.5)
        self.discriminator_optimizer.step()
        return dis_loss.item()

    def compute_loss(self, x_real, y_real):
        with torch.no_grad():
            zsc, zs = self.share_encoder(x_real)
            pred_y = self.classifier(zsc)
            _, pred = torch.max(pred_y.data, 1)
            total = y_real.size(0)
            correct = pred.eq(y_real.data).cpu().sum()
        return self.ce_loss(pred_y, y_real).item(), correct, total

    def compute_loss2(self, x_real, domain):
        with torch.no_grad():
            y_real = Variable(torch.Tensor(x_real.size(0)).fill_(domain), requires_grad=False).cuda()
            zsc, zs = self.share_encoder(x_real, with_argumentation=True)
            pred_y = self.discriminator2(zs)
            pred = (pred_y < 0.5).int().view(x_real.size(0))
            total = y_real.size(0)
            correct = pred.eq(y_real.data).cpu().sum()
        return 0.0, correct, total


if __name__ == '__main__':
    ball_loss_weight = 10.0
    discriminator_loss = 1.0

    torch.cuda.set_device(0)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.device = DEVICE
    args.x_dim = 9
    args.d_AE = 128
    args.n_class = 6
    args.seq_len = 128
    args.n_epoch = 351
    args.batch_size = 64
    args.now_model_name = 'MLP'
    for v1, v2 in zip([2, 26, 7, 16, 6, 7, 13, 16, 29, 13],
                      [4, 3, 25, 9, 23, 8, 7, 10, 14, 29]):
        data_train = U_data([v1])  # for one to one
        data_target = U_data([v2])
        train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(data_target, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(data_target, batch_size=args.batch_size, shuffle=False, drop_last=False)
        for i in range(1):
            result_total = []
            data_loader = train_loader
            for j in range(1):
                model = CATDModel(args)
                model = model.cuda()
                result, acc_all = [], []
                best_tar_f1, best_iter = 0.0, 0
                best_loss = 100.0
                change_center = False
                for i in range(args.n_epoch):
                    correct, total, loss = 0, 0, 0
                    correct2, total2, loss2 = 0, 0, 0
                    model.train()
                    pre_train_step = 100
                    for (xs, ys), (xt, yt) in zip(data_loader, test_loader):
                        xs = torch.squeeze(xs.to(torch.float32), -1)
                        xt = torch.squeeze(xt.to(torch.float32), -1)
                        ys = torch.as_tensor(ys,  dtype=torch.long).cuda()
                        yt = torch.as_tensor(yt,  dtype=torch.long).cuda()
                        xs, xt, ys, yt = xs.cuda(), xt.cuda(), ys.cuda(), yt.cuda()
                        model.train_step(xs, ys, xt, change_center=True if i <= pre_train_step else False,
                                         kl_w=ball_loss_weight if i > pre_train_step else 0.0,
                                         dis_w=discriminator_loss if i > pre_train_step else 0.1,
                                         with_argumentation=True)
                    if i == pre_train_step:
                        model.eval()
                        model.init_center_c(data_loader)
                        model.reset()
                    model.eval()
                    correct_1, total_1 = 0, 0
                    for ind, (xs, ys) in enumerate(data_loader):
                        xs = torch.squeeze(xs.to(torch.float32), -1)
                        ys = torch.as_tensor(ys, dtype=torch.long).cuda()
                        xs, ys = xs.cuda(), ys.cuda()
                        fake_loss = model.compute_loss(xs, ys)
                        correct += fake_loss[1]
                        total += fake_loss[2]
                        fake_loss_2 = model.compute_loss2(xs, 1)
                        correct_1 += fake_loss_2[1]
                        total_1 += fake_loss_2[2]
                        loss += fake_loss[0] * fake_loss[2]
                    for ind, (xs, ys) in enumerate(valid_loader):
                        xs = torch.squeeze(xs.to(torch.float32), -1)
                        ys = torch.as_tensor(ys, dtype=torch.long).cuda()
                        xs, ys = xs.cuda(), ys.cuda()
                        true_loss = model.compute_loss(xs, ys)
                        correct2 += true_loss[1]
                        total2 += true_loss[2]
                        fake_loss_2 = model.compute_loss2(xs, 0)
                        correct_1 += fake_loss_2[1]
                        total_1 += fake_loss_2[2]
                        loss2 += true_loss[0] * true_loss[2]
                    acc_all.append(float(correct2) / total2)
                    best_tar_f1 = max(acc_all)
                    best_iter = acc_all.index(best_tar_f1) + 1
                    result.append([i, loss / total, float(correct) / total, loss2 / total2, float(correct2) / total2])
                    result_np = np.array(result, dtype=float)
                    if loss2 / total2 < best_loss:
                        best_loss = loss2 / total2
                    print('%d, %d, Domain classification precision: %.6f' % (v1, v2, float(correct_1)/total_1))
                    print('Source %d, Target % d, Epoch %d, Target Loss: %.6f, Target precision: %.6f' %
                        (v1, v2, i, loss2 / total2, float(correct2) / total2))
                print('%d, %d, Best target precision: %.6f, Best loss: %.6f' % (v1, v2, best_tar_f1, best_loss))



