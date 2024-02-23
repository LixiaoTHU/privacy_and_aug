import time, datetime, shutil, os
import numpy as np
import argparse
import utils
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import math
import argparse
from sklearn.metrics import roc_curve

from dataset import get_loaders, root

import matplotlib.pyplot as plt

sample_num = 60000
f = open("sampleinfo/samplelist.txt", "r")
samplelist = eval(f.read())
f.close()

# sample_num = 5010
# f = open("sampleinfo/samplelist_locations.txt", "r")
# samplelist = eval(f.read())
# f.close()

# allmodellist = [i for i in range(0, 128)]
# f = open("sampleinfo/target.txt", "r")
# target = eval(f.read())
# f.close()
# target = set(target)
# for it in target:
#     y[it] += 1

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Eval privacy using 'First Principle'.")
    parser.add_argument('--dataset', default = 'cifar10', choices=["cifar10", "cifar100", "cifar10_resnet", "cifar100_resnet", "svhn", "locations", "purchase"])
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--s_model', default=10, type=int, help='s_model')
    parser.add_argument('--t_model', default=20, type=int, help='t_model')
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--aug_type', default="cutout", type=str, help='aug type')
    return parser.parse_args()

args = get_arguments()

def normal(mu, sigma2, x, multi = False):
    r = -np.log(sigma2) / 2 - (x - mu) ** 2 / (2 * sigma2)
    return np.sum(r)

def ROC_curve(y, pred, save = False, show = True, name = "test"):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    pred_sort = np.sort(pred)[::-1]
    index = np.argsort(pred)[::-1]
    y_sort = y[index]
    tpr = []
    fpr = []
    thr = []
    for i,item in enumerate(pred_sort):
        tpr.append(np.sum((y_sort[:i] == 1)) / pos)
        fpr.append(np.sum((y_sort[:i] == 0)) / neg)
        thr.append(item)
    for i in range(len(fpr)-1, -1, -1):
        if fpr[i] <= 1e-3:
#             print("TPR @ 0.1% FPR: ", (str(tpr[i] * 100)))
            tpr_0_1 = tpr[i] * 100
            break
    for i in range(len(fpr)-1, -1, -1):
        if fpr[i] <= 1e-4:
#             print("TPR @ 0.01% FPR: ", (str(tpr[i] * 100)))
            tpr_00_1 = tpr[i] * 100
            break
    for i in range(len(fpr)-1, -1, -1):
        if fpr[i] <= 1e-5:
#             print("TPR @ 0.001% FPR: ", (str(tpr[i] * 100)))
            tpr_000_1 = tpr[i] * 100
            break
    logfpr = np.log10(np.array(fpr) + 1e-5)
    logtpr = np.log10(np.array(tpr) + 1e-5) + 5
    # logfpr = np.array(fpr)
    # logtpr = np.array(tpr)
    eps = logfpr[1:] - logfpr[:-1]
    auroc = np.sum(eps * np.array(logtpr)[1:]) / (5 * 5)
#     print("AUROC: ", auroc)
    
    if show:
        plt.yscale('log')
        plt.xscale('log')
        plt.plot(fpr, tpr, 'k')
        plt.title('Receiver Operating Characteristic: %s' % (aug_type))
        plt.plot([(0,0),(1,1)],'r--')
        plt.xlim([1e-5,1])
        plt.ylim([1e-5,1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        if save:
            plt.savefig(name + ".pdf", dpi=300)
        else:
            plt.show()
    return tpr_0_1, tpr_00_1, tpr_000_1, auroc




if __name__ == "__main__":
    # aug_types = ["base", "smooth", "disturblabel", "noise", "cutout", "mixup", "jitter", "distillation", "pgdat", "trades", "AWP", "TradesAWP"]
    aug_types = ["cutout"]
    assert args.aug_type in aug_types

    info = dict()
    info_10 = dict()

    dataset = args.dataset
    if args.multi:
        foldername = "phy_multi"
    else:
        foldername = "phy"
    for aug_i in range(len(aug_types)):
        al, bl, cl, dl, tl, tl_conf = [], [], [], [], [], []
        for trial in range(args.s_model, args.t_model):
            
            allmodellist = [i for i in range(0, 128)]
            allmodellist.remove(trial)
            target = set(samplelist[trial])
            y = np.zeros(sample_num)
            y[samplelist[trial]] = 1

            
            aug_type = aug_types[aug_i]
            dirs = os.path.join(root, foldername, dataset, aug_type)
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
                npdict[index] = np.load("%s/%s_%s.npy" % (dirs, foldername, str(index)))


            in_dict = dict()
            out_dict = dict()
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
                if args.multi:
                    in_u, in_sigma = np.mean(confsin, axis = 0), np.var(confsin, axis = 0)
                    out_u, out_sigma = np.mean(confsout, axis = 0), np.var(confsout, axis = 0)
                else:
                    in_u, in_sigma = np.mean(confsin), np.var(confsin)
                    out_u, out_sigma = np.mean(confsout), np.var(confsout)
                in_dict[i] = (in_u, in_sigma)
                out_dict[i] = (out_u, out_sigma)

            base = np.load("%s/%s_%s.npy" % (dirs, foldername, str(trial)))
            baseeval= []
            for i in range(sample_num):
                baseeval.append(normal(in_dict[i][0], in_dict[i][1], base[i]) -  normal(out_dict[i][0], out_dict[i][1], base[i]))

            tpr_0_1, tpr_00_1, tpr_000_1, auroc = ROC_curve(y, baseeval, show = False)
            baseeval = np.array(baseeval)

            fpr, tpr, thr = roc_curve(y, baseeval)
            AccList = 1 - np.logical_xor(baseeval[:, np.newaxis] > thr[np.newaxis, :], y[:, np.newaxis]).sum(0) / len(baseeval)
            Acc_opt = np.max(AccList)
            tl.append(Acc_opt)
            al.append(tpr_0_1)
            bl.append(tpr_00_1)
            cl.append(tpr_000_1)
            dl.append(auroc)

            
            scores = 1 / (1 + np.exp(-base))
            if args.multi:
                scores = scores.mean(1)
            fpr, tpr, thr = roc_curve(y, scores)
            AccList = 1 - np.logical_xor(scores[:, np.newaxis] > thr[np.newaxis, :], y[:, np.newaxis]).sum(0) / len(scores)
            Acc_opt = np.max(AccList)
            tl_conf.append(Acc_opt)


        print('####################')
        print(aug_i)
        print("TPR @ 0.1% FPR: ", (str(np.mean(al))))
        print(np.std(al))
        print("TPR @ 0.001% FPR: ", (str(np.mean(cl))))
        print(np.std(cl))
        print("AUROC: ", str(np.mean(dl)))
        print(np.std(dl))
        print("Balanced accuracy: ", str(np.mean(tl)))
        print(np.std(tl))
        print("Balanced accuracy of confidence attack: ", str(np.mean(tl_conf)))
        info[aug_types[aug_i]] = (np.mean(al), np.std(al), np.mean(cl), 
                        np.std(cl), np.mean(dl), np.std(dl), 
                                    np.mean(tl), np.std(tl))
        info_10[aug_types[aug_i]] = (al, cl, dl, tl, tl_conf)
    if args.save_results:
        name = dataset + "_multi_10_info.txt" if args.multi else dataset + "_10_info.txt"
        f = open(name, "w")
        f.write(str(info_10))
        f.close()
        name = dataset + "_multi_info.txt" if args.multi else dataset + "_info.txt"
        f = open(name, "w")
        f.write(str(info))
        f.close()
    print(info)
        