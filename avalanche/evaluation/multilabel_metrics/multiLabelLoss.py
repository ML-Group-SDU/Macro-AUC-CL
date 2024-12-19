import numpy as np
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss, MultiLabelSoftMarginLoss
from itertools import product


class NaiveReweightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCEWithLogitsLoss()

    def forward(self, pred_y, true_y):  # pred_y �� true_y ����������ʽ
        # ��ÿһ���࣬�ҳ�������ص��������Լ������޹ص���������صģ�true��Ϊ1������أ�true��Ϊ 0

        device = true_y.device
        losses = []
        for c in range(true_y.shape[1]):
            c = torch.tensor([c]).to(device)
            y_true_c = true_y[:, c]
            y_pred_c = pred_y[:, c]
            if len(torch.unique(y_true_c)) == 2:
                loss = self.single_class_loss(y_true_c, y_pred_c)
                losses.append(loss.to(device))

        # if len(losses) == 0: # ȫ��0����1���޷���reweight loss��  ����ת��������Ӹ�bce loss
        #     print(true_y.shape)
        #     # c_idxs = torch.stack(list(c_nums.keys()))
        #     for c,num in c_nums.items():
        #         y_true_c = true_y[:, c]
        #         y_pred_c = pred_y[:, c]
        #         loss_ = self.bce(y_true_c,y_pred_c)
        #         losses.append(loss_.to(device))

        losses = torch.mean(torch.stack(losses))

        return losses

    def single_class_loss(self, y_true_c, y_pred_c):
        y_true_c.reshape(1, -1)
        pos = [i for i in range(y_true_c.shape[0]) if y_true_c[i] == 1]
        neg = [i for i in range(y_true_c.shape[0]) if y_true_c[i] == 0]

        poss = y_pred_c[pos]
        negs = y_pred_c[neg]
        repeted_poss = torch.repeat_interleave(poss, len(neg), 0)
        repeted_negs = negs.repeat(len(pos), 1)

        return torch.div(torch.sum(torch.add(self.loss(repeted_poss), self.loss(-repeted_negs))),
                         (len(pos) * len(neg)))

    def loss(self, ele):
        return torch.log(torch.add(1, torch.exp(-ele)))


class ReweightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.bce = BCEWithLogitsLoss()

    def forward(self, pred_y, true_y, c_nums):  # pred_y �� true_y ����������ʽ
        # ��ÿһ���࣬�ҳ�������ص��������Լ������޹ص���������صģ�true��Ϊ1������أ�true��Ϊ 0
        device = true_y.device
        losses = []
        for c, num in c_nums.items():
            c = torch.tensor([c]).to(device)
            y_true_c = true_y[:, c]
            y_pred_c = pred_y[:, c]
            if len(torch.unique(y_true_c)) == 2:
                loss = self.single_class_loss(y_true_c, y_pred_c)
                losses.append(loss.to(device))

        if len(losses) == 0:  # ȫ��0����1���޷���reweight loss��  ����ת��������Ӹ�bce loss
            print(true_y.shape)
            # c_idxs = torch.stack(list(c_nums.keys()))
            # for c,num in c_nums.items():
            #     y_true_c = true_y[:, c]
            #     y_pred_c = pred_y[:, c]
            #     loss_ = self.bce(y_true_c,y_pred_c)
            #     losses.append(loss_.to(device))

        losses = torch.mean(torch.stack(losses))

        return losses

    def single_class_loss(self, y_true_c, y_pred_c):
        y_true_c.reshape(1, -1)
        pos = [i for i in range(y_true_c.shape[0]) if y_true_c[i] == 1]
        neg = [i for i in range(y_true_c.shape[0]) if y_true_c[i] == 0]

        poss = y_pred_c[pos]
        negs = y_pred_c[neg]
        repeted_poss = torch.repeat_interleave(poss, len(neg), 0)
        repeted_negs = negs.repeat(len(pos), 1)

        return torch.div(torch.sum(torch.add(self.loss(repeted_poss), self.loss(-repeted_negs))),
                         (len(pos) * len(neg)))

    def loss(self, ele):
        return torch.log(torch.add(1, torch.exp(-ele)))


class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_y, true_y, c_nums, C):  # pred_y �� true_y ����������ʽ
        # ��ÿһ���࣬�ҳ�������ص��������Լ������޹ص���������صģ�true��Ϊ1������أ�true��Ϊ 0

        device = true_y.device
        losses = []
        for c, num in c_nums.items():
            c = torch.tensor([c]).to(device)
            y_true_c = true_y[:, c]
            y_pred_c = pred_y[:, c]
            if len(torch.unique(y_true_c)) == 2:
                loss = self.single_class_loss(y_true_c, y_pred_c, num, C)
                losses.append(loss.to(device))

        if len(losses) == 0:
            print(true_y.shape)
        losses = torch.mean(torch.stack(losses))

        return losses

    def single_class_loss(self, y_true_c, y_pred_c, c_num, C):
        # y_true_c.reshape(1,-1)
        # y_hat:Ԥ���ǩ���Ѿ���sigmoid/softmax���� shape is (batch_size, 1)
        # y����ʵ��ǩ��һ��Ϊ0��1�� shape is (batch_size)
        y_true_c = y_true_c.type(torch.int64)
        hyper_c = C
        delta_pos = hyper_c / torch.pow(c_num[0], 1 / 4)
        delta_neg = hyper_c / torch.pow(c_num[1], 1 / 4)
        y_pred_c = y_pred_c - delta_pos
        y_pred_c = torch.sigmoid(y_pred_c)

        # y_hat = torch.cat((torch.sub(1 - y_pred_c,delta_neg), torch.sub(y_pred_c,delta_pos)), 1)  # ����������ĸ��ʶ��г���y_hat��״��Ϊ(batch_size, 2)
        y_hat = torch.cat((1 - y_pred_c, y_pred_c), 1)  # ����������ĸ��ʶ��г���y_hat��״��Ϊ(batch_size, 2)

        # ����y�궨����ʵ��ǩ��ȡ��Ԥ��ĸ��ʣ���������ʧ
        b = y_true_c.view(-1, 1)
        a = y_hat.gather(1, b)
        res = - torch.log(a).mean()
        return res
        # ��������loss��ֵ


class ReweightedMarginLoss(nn.Module):
    def __init__(self, C, nth_power):
        super().__init__()
        self.nth_power = nth_power
        self.hyper_c = C

    def forward(self, pred_y, true_y, c_nums):  # pred_y �� true_y ����������ʽ
        # ��ÿһ���࣬�ҳ�������ص��������Լ������޹ص���������صģ�true��Ϊ1������أ�true��Ϊ 0
        device = true_y.device
        losses = []
        for c, num in c_nums.items():
            c = torch.tensor([c]).to(device)
            y_true_c = true_y[:, c]
            y_pred_c = pred_y[:, c]
            if len(torch.unique(y_true_c)) == 2:
                loss = self.single_class_loss(y_true_c, y_pred_c, num)
                losses.append(loss.to(device))

        if len(losses) == 0:
            print(true_y.shape)
        losses = torch.mean(torch.stack(losses))

        return losses

    def single_class_loss(self, y_true_c, y_pred_c, c_num):
        y_true_c.reshape(1, -1)
        pos = [i for i in range(y_true_c.shape[0]) if y_true_c[i] == 1]
        neg = [i for i in range(y_true_c.shape[0]) if y_true_c[i] == 0]

        poss = y_pred_c[pos]
        negs = y_pred_c[neg]
        repeted_poss = torch.repeat_interleave(poss, len(neg), 0)
        repeted_negs = negs.repeat(len(pos), 1)

        delta_pos = self.hyper_c / torch.pow(c_num[0], self.nth_power)
        delta_neg = self.hyper_c / torch.pow(c_num[1], self.nth_power)

        # return torch.div(torch.sum(torch.add(self.loss(torch.sub(repeted_poss,delta_pos)),
        #                                      self.loss(torch.sub(-repeted_negs,delta_neg)))), (len(pos)*len(neg)))
        return torch.div(torch.sum(torch.add(self.loss(torch.sub(repeted_poss, delta_pos)),
                                             self.loss(torch.sub(-repeted_negs, delta_pos)))), (len(pos) * len(neg)))

        # else:
        #     return 0
        # elif len(pos)==0 and len(neg)!=0: # true label ��Ϊ 0
        #     negs = y_pred_c[neg]
        #     return torch.div(torch.sum(self.loss(-negs)),(len(neg)))
        # elif len(pos)!=0 and len(neg)==0:
        #     poss = y_pred_c[pos]
        #     return torch.div(torch.sum(self.loss(poss)),(len(pos)))

    def loss(self, ele):
        return torch.log(torch.add(1, torch.exp(-ele)))


class SpecficBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = BCEWithLogitsLoss()

    def forward(self, pred_y, true_y):  # pred_y �� true_y ����������ʽ
        # true_y,pred_y = eliminate_all_zero_columns(y_true=true_y,y_pred=pred_y)
        device = true_y.device
        n_classes = true_y.shape[1]
        y_trues = []
        y_preds = []
        # indicators = torch.sum(true_y,1)
        # torch.where()
        for c in range(n_classes):
            class_indx = torch.tensor([c]).to(device)
            y_true_c = true_y[:, class_indx]
            y_pred_c = pred_y[:, class_indx]
            if len(torch.unique(y_true_c)) == 2 or torch.unique(y_true_c) == 1:  # ������ǰ�����漰�����
                y_trues.append(y_true_c)
                y_preds.append(y_pred_c)

        if len(y_trues) > 0:
            true_y = torch.concat(y_trues, dim=1)
            pred_y = torch.concat(y_preds, dim=1)
            loss = self.bce(pred_y, true_y)
            return loss
        else:
            return 0


class MultiSoftLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.msl = MultiLabelSoftMarginLoss()

    def forward(self, pred_y, true_y):  # pred_y �� true_y ����������ʽ
        # true_y,pred_y = eliminate_all_zero_columns(y_true=true_y,y_pred=pred_y)
        device = true_y.device
        n_classes = true_y.shape[1]
        y_trues = []
        y_preds = []
        # indicators = torch.sum(true_y,1)
        # torch.where()
        for c in range(n_classes):
            class_indx = torch.tensor([c]).to(device)
            y_true_c = true_y[:, class_indx]
            y_pred_c = pred_y[:, class_indx]
            if len(torch.unique(y_true_c)) == 2 or torch.unique(y_true_c) == 1:  # ������ǰ�����漰�����
                y_trues.append(y_true_c)
                y_preds.append(y_pred_c)

        if len(y_trues) > 0:
            true_y = torch.concat(y_trues, dim=1)
            pred_y = torch.concat(y_preds, dim=1)
            loss = self.msl(pred_y, true_y)
            return loss
        else:
            return 0


class ForMicroLoss(nn.Module):
    def __init__(self, mode="u_1"):
        super().__init__()
        self.mode = mode

    def forward(self, pred_y, true_y):  # pred_y �� true_y ����������ʽ
        device = true_y.device
        losses = []
        K = true_y.shape[-1]
        n = true_y.shape[0]
        S_pos = torch.where(true_y == 1)
        S_neg = torch.where(true_y == 0)

        if self.mode == "u_1":
            for i in range(K):
                for j in range(K):
                    tmp = 0
                    S_i_pos = torch.where(S_pos[1] == i)[0].tolist()
                    S_j_neg = torch.where(S_neg[1] == j)[0].tolist()
                    elems = product(*[S_i_pos, S_j_neg])
                    for elem in elems:
                        print(elem)
                        tmp += len(S_pos) * self.loss(pred_y[p[0], i]) + len(S_neg) * self.loss(pred_y[q[0], j])
                    losses.append(tmp)
            l = torch.sum(torch.stack(losses)) / (len(S_pos) * len(S_neg) * n * K)

        elif self.mode == "u_2":
            for i in range(K):
                for a in range(n):
                    e1 = (self.loss(pred_y[a, i]) / len(S_pos)) if (i, a) in S_pos else 0
                    e2 = (self.loss(-pred_y[a, i]) / len(S_neg)) if (i, a) in S_neg else 0
                    losses.append(e1 + e2)
            l = torch.sum(torch.stack(losses))
        else:
            raise NotImplementedError

        return l

    def loss(self, ele):
        return torch.log(torch.add(1, torch.exp(-ele)))

def choose_loss(args):
    print(args.crit)
    if args.crit == "BCE": # BCE without or with WRS
        crit = SpecficBCELoss()
    elif args.crit == "R":
        crit = ReweightedLoss()
    elif args.crit == "M":
        crit = MarginLoss()
    elif args.crit == "RM":
        crit = ReweightedMarginLoss(C=args.C,nth_power=args.nth_power)
    else:
        raise NotImplementedError
    return crit