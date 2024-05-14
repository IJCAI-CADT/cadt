from torch.autograd import Variable
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.networks import CNNEncoder,Discriminator,Classifier

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
