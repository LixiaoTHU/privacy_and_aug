import os
import sys
sys.path.append('../')

from sklearn.manifold import TSNE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import roc_auc_score as roc_auc

def test(net, loader, device, attacker=None, adv_args=None, defender=None, defender_info=None, display=True, target_net=None, **kwargs):
    # net.eval()
    if target_net is None:
        target_net = net
    total_loss = []
    total_accuracy = []
    total_success = []
    total_num = 0

    if display:
        loader = tqdm(loader, total=len(loader), position=0)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        total_num += inputs.shape[0]
        if attacker is None:
            num_class = 10
            targets = (labels + labels.new(labels.size()).random_(1, num_class)).remainder(num_class)
            inputs_adv = inputs
        else:
            inputs_adv, targets = attacker(target_net, inputs, labels, **adv_args, **kwargs)

        with torch.no_grad():
            if defender_info is not None:
                pred = defender_info(inputs, inputs_adv)
                loss = pred.new_zeros([])
            elif defender is not None:
                pred = defender(inputs_adv)
                loss = pred.new_zeros([])
            else:
                outputs = net(inputs_adv)
                pred = outputs.data.max(1)[1]
                loss = F.cross_entropy(outputs, labels, reduction='sum')
            total_loss.append(loss.item())
            total_accuracy.append(pred.eq(labels.data).float().sum().item())
            total_success.append(pred.eq(targets.data).float().sum().item())

    avg_test_loss = np.sum(total_loss) / total_num
    avg_test_acc = np.sum(total_accuracy) / total_num * 100
    avg_test_suc = np.sum(total_success) / total_num * 100
    return avg_test_loss, avg_test_acc, avg_test_suc


root = os.path.join(os.environ['HOME'], 'data')
cache_dir = os.path.join(root, 'cache', 'vae')


def fake_inputs(inputs, disc_net, eps, eps_range, clip_min, clip_max, **kwargs):
    B = inputs.shape[0]
    if eps_range is not None:
        eps = (inputs.new_empty([B, 1, 1, 1]).uniform_() * eps * eps_range)
    inputs_f = inputs + inputs.new_empty(inputs.size()).normal_() * eps
    inputs_f.clamp_(clip_min, clip_max)
    inputs_p = torch.cat([inputs, inputs_f], 0)
    labels_p = torch.cat([inputs.new_zeros([B, 1]), inputs.new_ones([B, 1])], 0)
    return inputs_p, labels_p


def computeMetrics(scores, FPR):
    scores1, scores0 = scores

    labels0 = np.zeros_like(scores0)
    labels1 = np.ones_like(scores1)

    scores = np.concatenate((scores0, scores1))
    labels = np.concatenate((labels0, labels1))

    # ROC curve
    fpr, tpr, thr = roc_curve(labels, scores)
    TPR = np.interp(FPR, fpr, tpr)

    ths = [1e-3, 1e-4, 1e-5]
    TPRs_lowFPR_interp = np.interp(ths, fpr, tpr)
    TPRs_lowFPR = [tpr[(fpr <= th).nonzero()[0][-1]] for th in ths]
    TPRs_lowFPR = np.array(TPRs_lowFPR)

    # FPR @TPR95
    FPRs = np.interp([.80, .85, .90, .95], tpr, fpr)

    # AUROC
    AUROC = roc_auc(labels, scores)

    # Optimal Accuracy
    # AccList = [accuracy_score(scores > t, labels) for t in thr]
    AccList = 1 - np.logical_xor(scores[:, np.newaxis] > thr[np.newaxis, :], labels[:, np.newaxis]).sum(0) / len(scores)
    Acc_opt = np.max(AccList)
    ind = np.argmax(AccList)
    thr_opt = thr[ind]

    metrics = {
        'AUROC': AUROC,
        'acc_list': AccList,
        'acc_opt': Acc_opt,
        'thr_opt': thr_opt,
        'fpr': fpr,
        'tpr': tpr,
        'thr': thr,
        'FPRs_4': FPRs,
        'TPRs_lowFPR_interp': TPRs_lowFPR_interp,
        'TPRs_lowFPR': TPRs_lowFPR,
        'FPR': FPR,
        'TPR': TPR,
    }
    return metrics


def normal(mu, sigma2, x):
    r = - np.log(sigma2) / 2 - (x - mu) ** 2 / (2 * sigma2)
    return r


def compute_feature(root, aug_type, trial, dataset, samplelist, sample_num=60000):
    dirs = os.path.join(root, "phy", dataset, aug_type)
    allmodellist = list(range(0, 128))
    allmodellist.remove(trial)

    IN = []
    OUT = []
    for i in range(sample_num):
        IN.append([])
        OUT.append([])
    alls = [i for i in range(sample_num)]
    for index in allmodellist:
        slist = samplelist[index]
        for it in slist:
            IN[it].append(index)
        outslist = set(alls) - set(slist)
        for it in outslist:
            OUT[it].append(index)

    npdict = dict()
    for index in allmodellist:
        npdict[index] = np.load("%s/phy_%s.npy" % (dirs, str(index)))

    print('computing mean & var for in & out')
    in_dict = dict()
    out_dict = dict()
    confs_dict = dict()
    for i in range(sample_num):
        confsin, confsout = [], []
        for it in IN[i]:
            x = npdict[it][i]
            confsin.append(x)
        for it in OUT[i]:
            x = npdict[it][i]
            confsout.append(x)
        confsin = np.array(confsin)
        confsout = np.array(confsout)
        in_u, in_sigma = np.mean(confsin), np.var(confsin)
        out_u, out_sigma = np.mean(confsout), np.var(confsout)
        in_dict[i] = (in_u, in_sigma)
        out_dict[i] = (out_u, out_sigma)
        confs_dict[i] = (confsin, confsout)

    print('test one model')
    base = np.load("%s/phy_%s.npy" % (dirs, str(trial)))
    base_in = []
    base_out = []
    # baseeval_offline = []
    for i in range(sample_num):
        a = normal(in_dict[i][0], in_dict[i][1], base[i])
        b = normal(out_dict[i][0], out_dict[i][1], base[i])
        base_in.append(a)
        base_out.append(b)
        # baseeval_offline.append((base[i] - out_dict[i][0]) / np.sqrt(out_dict[i][1]))
    base_in = np.array(base_in)
    base_out = np.array(base_out)
    baseeval = base_in - base_out
    # baseeval_offline = np.array(baseeval_offline)
    return base, baseeval, confs_dict
