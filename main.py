import sys
import argparse
import numpy as np
import torch
from dataset.dataset import U_data,Wisdm_ar_data
from torch.utils.data import DataLoader
from model.CADT import CATDModel


parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--x_dim", type=int, default=9, help="time_series_dim")
parser.add_argument("--d_AE", type=int, default=128, help="AE_dim")
parser.add_argument("--n_class", type=int, default=6, help="n_class")
parser.add_argument("--seq_len", type=int, default=128, help="seq_len")
parser.add_argument("--n_epoch", type=int, default=351, help="epochs")
parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
parser.add_argument("--device", type=int, default=0, help="cuda index")
parser.add_argument("--ball_loss_weight", type=int, default=10.0, help="part_of_loss")
parser.add_argument("--discriminator_loss", type=int, default=1.0, help="part_of loss")
parser.add_argument("--source_domain", type=int, default=2, help="0-30")
parser.add_argument("--target_domain", type=int, default=4, help="0-30")
parser.add_argument("--dataset", type=str, default='ucihar', help="ucihar or wisdm")
args = parser.parse_args()

if __name__ == '__main__':

    torch.cuda.set_device(args.device)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v1 = args.source_domain
    v2 = args.target_domain
    if args.dataset == 'ucihar':
        data_train = U_data([v1])  # for one to one
        data_target = U_data([v2])
    elif args.dataset == 'wisdm':
        data_train = Wisdm_ar_data([v1])  # for one to one
        data_target = Wisdm_ar_data([v2])
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
                                     kl_w=args.ball_loss_weight if i > pre_train_step else 0.0,
                                     dis_w=args.discriminator_loss if i > pre_train_step else 0.1,
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