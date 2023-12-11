import numpy as np

import os, json, time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import logging
import copy
import datetime

import utils
import random
from models import *
from dataset import get_loaders, root
from thop import profile
from advtrain import cal_adv
from trades_awp import AdvWeightPerturb, TradesAWP

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training models.")
    parser.add_argument('--exp_name', type=str, default=root)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--load_best_model', action='store_true', default=False)
    parser.add_argument('--data_parallel', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--dataset', default = 'cifar10', choices=["cifar10", "cifar100", "svhn", "purchase", "locations"])
    # parser.add_argument('--awp-gamma', default=0.005, type=float,
    #                 help='whether or not to add parametric noise')

    parser.add_argument('--epsilon', default=8, type=int, help='perturbation')

    parser.add_argument('--s_model', default=0, type=int, help='s_model')
    parser.add_argument('--t_model', default=16, type=int, help='t_model')
    parser.add_argument('--aug_type', default="pgdat", type=str, help='aug type')
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--mode', default="train", choices=["all", "train", "target", "eval"])

    parser.add_argument('--use_random_smooth', action='store_true', default=False)
    parser.add_argument('--without_base', action='store_true', default=False)
    parser.add_argument('--suffix', default="")
    parser.add_argument('--global_mode', default="train", type=str, choices=["train", "gap", "decide", "adv"])
    parser.add_argument('--cnn', action='store_true', default=False)
    return parser.parse_args()

args = get_arguments()

def adjust_learning_rate(optimizer, epoch, allepoch=100):
    if allepoch == 50:
        if epoch >= 0.5 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 0.75 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.01
    elif allepoch == 100:
        if epoch >= 0.75 * allepoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
        elif epoch >= 0.9 * allepoch:
            for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * 0.01
    else:
        exit(0)

def SoftLabelNLL(predicted, target, reduce=False):
    if reduce:
        return -(target * predicted).sum(dim=1).mean()
    else:
        return -(target * predicted).sum(dim=1)

def train(epoch, model, optimizer, trainloader, ENV, aug_type, criterion, cfg, teacher, logger, awp_adversary, aug_index = 0):
    logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
    model.train()
    log_frequency = 50
    category_loss= 0
    category_correct = 0
    category_total = 0
    criterion_kl = nn.KLDivLoss(reduction='sum')
    if aug_type == "mixup":
        iterator = zip(trainloader, trainloader)
    else:
        iterator = enumerate(trainloader)
    for batch in iterator:
        start = time.time()
        if aug_type == "mixup":
            (imgs, cids), (imgs_2, cids_2) = batch
            imgs, cids = imgs.to(device), cids.to(device)
            imgs_2, cids_2 = imgs_2.to(device), cids_2.to(device)
        else:
            _, (imgs, cids) = batch
            imgs, cids = imgs.to(device), cids.to(device)
        optimizer.zero_grad()

        # import cv2
        # cv2.imwrite("test.png", imgs[0].cpu().numpy().transpose(1,2,0)*255)
        # print(imgs[0,0,:20,:20])
        # print(imgs.shape)
        # print(cids.shape)
        # print(cids[0])
        # exit(0)
        
        if imgs.shape[0] == 1:
            continue

        if aug_type == "distillation":
            T = cfg['augmentation_params']['distillation'][aug_index]
            dt = teacher.forward_w_temperature(imgs, T).detach()
            pred = model(imgs)
            loss = criterion(pred, dt)
        elif aug_type == "smooth":
            pred = model(imgs)
            b_y_one_hot = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device = device).scatter_(1, cids.view(-1, 1), 1)

            smoothing_coef = cfg['augmentation_params']['smooth'][aug_index]
            if args.use_random_smooth:
                aux = torch.zeros(b_y_one_hot.shape).to(b_y_one_hot.device)
                aux[:,:10] += 1
                aux=aux[:,torch.randperm(aux.size()[1])]
                b_y_one_hot = (1-smoothing_coef) * b_y_one_hot + (smoothing_coef/(model.num_classes / 10)) * aux
            else:
                b_y_one_hot = (1-smoothing_coef) * b_y_one_hot + (smoothing_coef/model.num_classes)
            
            loss = criterion(pred, b_y_one_hot)
        elif aug_type == "mixup":
            alpha = cfg['augmentation_params']['mixup'][aug_index]
            lam = np.random.beta(alpha, alpha)
            b_x = (lam * imgs) + ((1 - lam) * imgs_2)
            b_y_one_hot = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device = device).scatter_(1, cids.view(-1, 1), 1)
            b_y_one_hot_2 = torch.zeros(imgs.shape[0], model.num_classes, dtype=torch.float, device = device).scatter_(1, cids_2.view(-1, 1), 1)

            b_cid = (lam * b_y_one_hot) + ((1 - lam) * b_y_one_hot_2)
            pred = model(b_x)
            loss = criterion(pred, b_cid)
        elif aug_type == "disturblabel":
            C = model.num_classes
            alpha = cfg['augmentation_params']['disturblabel'][aug_index]
            p_c = (1 - ((C - 1)/C) * alpha)
            p_i = (1 / C) * alpha
            b_y = cids.view(-1, 1)   # batch y

            b_y_one_hot = (torch.ones(b_y.shape[0], C) * p_i).to(device)
            b_y_one_hot.scatter_(1, b_y, p_c)
            b_y_one_hot = b_y_one_hot.view( *(tuple(cids.shape) + (-1,) ) )

            # sample from Multinoulli distribution
            distribution = torch.distributions.OneHotCategorical(b_y_one_hot)
            b_y_disturbed = distribution.sample()
            b_y_disturbed = b_y_disturbed.max(dim=1)[1]  # back to categorical
            pred = model(imgs)
            loss = criterion(pred, b_y_disturbed)
        elif aug_type == "01flip":
            B = imgs.shape[0]
            imgs = imgs.reshape(-1)
            percent = cfg['augmentation_params']['01flip'][aug_index]
            num_elements = int(imgs.numel() * percent)
            flip_indices = torch.randperm(imgs.numel())[:num_elements]
            imgs[flip_indices] = 1 - imgs[flip_indices]
            imgs = imgs.reshape(B, -1)
            pred = model(imgs)
            loss = criterion(pred, cids)
        elif aug_type == "dandd":
            
            C = model.num_classes
            if args.dataset == "cifar10":
                alpha = 0.05
            elif args.dataset == "cifar100":
                alpha = 0.3
            elif args.dataset == "svhn":
                print("Not prepared yet!")
                exit(0)
            p_c = (1 - ((C - 1)/C) * alpha)
            p_i = (1 / C) * alpha
            b_y = cids.view(-1, 1)   # batch y

            b_y_one_hot = (torch.ones(b_y.shape[0], C) * p_i).to(device)
            b_y_one_hot.scatter_(1, b_y, p_c)
            b_y_one_hot = b_y_one_hot.view( *(tuple(cids.shape) + (-1,) ) )

            # sample from Multinoulli distribution
            distribution = torch.distributions.OneHotCategorical(b_y_one_hot)
            b_y_disturbed = distribution.sample()
            b_y_disturbed = b_y_disturbed.max(dim=1)[1]  # back to categorical
            
            b_y_disturbed = b_y_disturbed.unsqueeze(0)
            gcids = cids.unsqueeze(0)

            T = 3
            dt = teacher.forward_w_temperature(imgs, T).detach()
            gg = dt.clone()

            forindex = torch.arange(dt.shape[0]).to(dt.device)
            logit_dis = dt[forindex, b_y_disturbed].clone()
            dt[forindex, b_y_disturbed] = dt[forindex, gcids].clone()
            dt[forindex, gcids] = logit_dis

            
            pred = model(imgs)
            loss = criterion(pred, dt)
            
        elif aug_type == "trades":
            imgs_adv = cal_adv(model, criterion, aug_type, imgs, cids, eps = args.epsilon)
            model.train()
            pred, logits = model(imgs, require_logits = True)
            loss_natural = criterion(pred, cids)
            pred_adv = model(imgs_adv)
            loss_robust = (1.0 / (pred.shape[0])) * criterion_kl(pred_adv, F.softmax(logits, dim = 1))
            loss_natural = criterion(pred, cids)
            loss = loss_natural + 6 * loss_robust
        elif aug_type ==  "pgdat":
            # import cv2
            # print(imgs.shape)
            # print(cids[2])
            # cv2.imwrite("test_ori.png", imgs[2].cpu().numpy().transpose(1,2,0)*255)
            imgs_adv = cal_adv(model, criterion, aug_type, imgs, cids, eps = args.epsilon)
            # cv2.imwrite("test.png", imgs_adv[2].cpu().numpy().transpose(1,2,0)*255)
            # exit(0)

            model.train()
            pred = model(imgs_adv)
            loss = criterion(pred, cids)
        elif aug_type == "AWP":
            imgs_adv = cal_adv(model, criterion, aug_type, imgs, cids, eps = args.epsilon)
            model.train()
            awp = awp_adversary.calc_awp(inputs_adv=imgs_adv,
                                        targets=cids)
            awp_adversary.perturb(awp)
            pred = model(imgs_adv)
            loss = criterion(pred, cids)
        elif aug_type == "TradesAWP":
            imgs_adv = cal_adv(model, criterion, aug_type, imgs, cids, eps = args.epsilon)
            model.train()
            awp = awp_adversary.calc_awp(inputs_adv=imgs_adv,
                                        inputs_clean=imgs,
                                        targets=cids,
                                        beta=6)
            awp_adversary.perturb(awp)

            pred, logits = model(imgs, require_logits = True)
            loss_natural = criterion(pred, cids)
            pred_adv = model(imgs_adv)
            loss_robust = (1.0 / (pred.shape[0])) * criterion_kl(pred_adv, F.softmax(logits, dim = 1))
            loss_natural = criterion(pred, cids)
            loss = loss_natural + 6 * loss_robust
        else:
            pred = model(imgs)
            loss = criterion(pred, cids)

        loss.backward()
        optimizer.step()
        
        category_loss += loss.item()
        _, predicted = pred.max(1)
        category_total += cids.size(0)
        category_correct += predicted.eq(cids).sum().item()

        if awp_adversary is not None:
            awp_adversary.restore(awp)

        end = time.time()
        time_used = end - start
        
        if ENV["global_step"] % log_frequency == 0:
            log_payload = {"loss": category_loss/category_total, "acc": 100.*(category_correct/category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                           **log_payload)
            logger.info(display)

        ENV["global_step"] += 1
    return category_loss/category_total, 100.*(category_loss/category_total)



def test(epoch, model, testloader, criterion, ENV, logger):
    logger.info("="*20 + "Test Epoch %d" % (epoch) + "="*20)
    model.eval()
    category_loss= 0
    category_correct = 0
    category_total = 0
    log_frequency = 50
    to_save = []
    for batch_idx, batch in enumerate(testloader):
        start = time.time()
        imgs, cids = batch
        imgs, cids = imgs.to(device), cids.to(device)

        if args.save_results:
            with torch.no_grad():
                pred = model.base_forward(imgs)
            logits = F.softmax(pred, dim=1)
            iss = torch.arange(pred.shape[0])
            phy = torch.log(logits[iss,cids[iss]])
            pred = F.log_softmax(logits, dim = 1)
            logits[iss,cids[iss]] = 0
            phy = phy - torch.log(torch.sum(logits,dim=1) + 1e-20)
            phy = phy.cpu().numpy()
            to_save.append(phy)
        else:
            with torch.no_grad():
                pred = model(imgs)

                # tcriterion = nn.NLLLoss()
                # imgs = cal_adv(model, tcriterion, "pgdat", imgs, cids, eps = 8)
                # model.eval()
                # pred = model(imgs)
                # loss = criterion(pred, cids)

        loss = criterion(pred, cids)

        _, predicted = pred.max(1)


        category_total += cids.size(0)
        category_loss += loss.item()
        category_correct += predicted.eq(cids).sum().item()

        end = time.time()
        time_used = end - start
        if (batch_idx+1) % log_frequency == 0:
            log_payload = {"category acc": 100.* (category_correct/category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                            **log_payload)
            logger.info(display)
    if args.save_results:
        phylist = np.concatenate(to_save)
        return 100.* (category_correct/category_total), category_loss / category_total, phylist

    return 100.* (category_correct/category_total), category_loss / category_total


def main(cfg, aug_type = "none", index = 0, aug_index = 0):
    if args.dataset == "cifar10":
        if args.cnn:
            model = CNN(num_classes=10, channels = 3).to(device)
        else:
            model = ResNet18().to(device)
    elif args.dataset == "cifar100":
        model = ResNet18(num_classes=100).to(device)
    elif args.dataset == "svhn":
        model = CNN(num_classes=10, channels = 3).to(device)
    elif args.dataset == "purchase":
        model = MLP(num_classes=100, size=600).to(device)
    elif args.dataset == "locations":
        model = MLP(num_classes=30, size=446).to(device)

    if aug_type == "TradesAWP" or "AWP":
        proxy = copy.deepcopy(model)

    assert(aug_type in cfg['training_augmentations'])
    if aug_type in ['distillation', 'smooth', 'mixup', 'dandd']:
        criterion = lambda pred, target: SoftLabelNLL(pred, target, reduce=True)
    else:
        criterion = nn.NLLLoss()
    test_criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


    if args.exp_name == '':
        new_exp_name = 'exp_' + datetime.datetime.now()
    else:
        if (aug_type == "trades" or aug_type == "pgdat") and args.epsilon < 8:
            new_exp_name = os.path.join(args.exp_name, args.dataset, aug_type + "_" + str(args.epsilon))
        else:
            new_exp_name = os.path.join(args.exp_name, args.dataset, aug_type)
        
        if args.without_base:
            new_exp_name = new_exp_name + "_none"
    
    if len(args.suffix) > 0:
        new_exp_name = os.path.join(new_exp_name, args.suffix)

    if args.mode == "all" or args.mode == "target":
        index = args.mode
    exp_path = os.path.join(new_exp_name, "resnet18_" + str(index))
    log_file_path = os.path.join(exp_path, "resnet18_" + str(index))
    checkpoint_path = exp_path
    utils.create_path(checkpoint_path)

    logger = utils.setup_logger(name="resnet18_" + str(index), log_file=log_file_path + ".log")
    # profile_inputs = (torch.randn([1, 3, 32, 32]).to(device),)
    # flops, params = profile(model, inputs=profile_inputs, verbose=False)
    # flops = flops / 1e9
    starting_epoch = 0

    # logger.info("param size = %fMB", utils.count_parameters_in_MB(model))
    # logger.info("flops: %.4fG" % flops)
    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    ENV = { 'global_step': 0,
            'best_acc': 0.0,
            'curren_acc': 0.0,
            'best_pgd_acc': 0.0}


    if args.load_model:
        checkpoint = utils.load_model(os.path.join(checkpoint_path, 'model'), model)
        starting_epoch = checkpoint['epoch'] + 1
    if args.load_best_model:
        checkpoint = utils.load_model(os.path.join(checkpoint_path, 'model_best'), model)
        starting_epoch = checkpoint['epoch'] + 1
    if aug_type == 'distillation':
        if args.dataset == "cifar10":
            if args.cnn:
                teacher = CNN(num_classes=10, channels = 3).to(device)
            else:
                teacher = ResNet18().to(device)
        elif args.dataset == "cifar100":
            teacher = ResNet18(num_classes=100).to(device)
        elif args.dataset == "svhn":
            teacher = CNN(num_classes=10).to(device)
        elif args.dataset == "purchase":
            teacher = MLP(num_classes=100, size=600).to(device)
        elif args.dataset == "locations":
            teacher = MLP(num_classes=30, size=446).to(device)
        print(checkpoint_path.replace("distillation", "base"))
        tname = "none" if args.without_base else "base"
        utils.load_model(os.path.join(checkpoint_path.replace("distillation", tname), 'model'), teacher)
        teacher.eval()
    elif aug_type == "dandd":
        if args.dataset == "cifar10":
            if args.cnn:
                teacher = CNN(num_classes=10, channels = 3).to(device)
            else:
                teacher = ResNet18().to(device)
        elif args.dataset == "cifar100":
            teacher = ResNet18(num_classes=100).to(device)
        elif args.dataset == "svhn":
            # teacher = MLP(num_classes=10).to(device)
            teacher = CNN(num_classes=10).to(device)
        else:
            print("Not prepared yet!")
            exit(0)
        print(checkpoint_path.replace("dandd", "base"))
        tname = "none" if args.without_base else "base"
        utils.load_model(os.path.join(checkpoint_path.replace("dandd", tname), 'model'), teacher)
        teacher.eval()
        # return
    else:
        teacher = None
    if args.data_parallel:
        print('data_parallel')
        model = torch.nn.DataParallel(model).to(device)
        if aug_type == "TradesAWP" or "AWP":
            proxy = torch.nn.DataParallel(proxy).to(device)
    
    if aug_type == "TradesAWP":
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.005)
    elif aug_type == "AWP":
        proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
        awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.01)
        # awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=0.001)
    else:
        awp_adversary = None
    
    if args.mode == "eval":
        bs = cfg['regular_batch_size'] * 4
    else:
        bs = cfg['regular_batch_size']
    trainloader, testloader = get_loaders(args.dataset, aug_type, aug_index, cfg, 
                                            shuffle=True, batch_size=bs, 
                                            mode = args.mode, samplerindex = index,
                                            without_base = args.without_base)
    logger.info("Starting Epoch: %d" % (starting_epoch))
    if args.train:
        for epoch in range(starting_epoch, cfg['training_num_epochs']):
            adjust_learning_rate(optimizer, epoch, allepoch=cfg['training_num_epochs'])

            tc_acc, tc_loss = train(epoch, model, optimizer, trainloader, ENV, aug_type, criterion, cfg, teacher, logger, awp_adversary, aug_index = aug_index)
            vc_acc, vc_loss = test(epoch, model, testloader, test_criterion, ENV, logger)
            is_best = True if vc_acc > ENV['best_acc'] else False
            ENV['best_acc'] = max(ENV['best_acc'], vc_acc)
            ENV['curren_acc'] = vc_acc
            
            logger.info('Current loss: %.2f' % (vc_loss))
            logger.info('Current accuracy: %.2f' % (vc_acc))
            target_model = model.module if args.data_parallel else model
            utils.save_model(os.path.join(checkpoint_path, 'model'), target_model, epoch, save_best=is_best)
    else:
        if args.save_results:
            vc_acc, vc_loss, phylist = test(starting_epoch, model, testloader, test_criterion, ENV, logger)
            if (aug_type == "trades" or aug_type == "pgdat") and args.epsilon < 8:
                save_path = os.path.join(args.exp_name, "phy", args.dataset, aug_type + "_" + str(args.epsilon))
            else:
                save_path = os.path.join(args.exp_name, "phy", args.dataset, aug_type)
            utils.create_path(save_path)
            np.save(os.path.join(save_path, "phy_%s.npy" % (str(index))), phylist)
        else:
            tc_acc, tc_loss = test(starting_epoch, model, trainloader, test_criterion, ENV, logger)
            vc_acc, vc_loss = test(starting_epoch, model, testloader, test_criterion, ENV, logger)
        logger.info('Current loss: %.4f' % (vc_loss))
        logger.info('Current accuracy: %.2f' % (vc_acc))
    # logging.shutdown()
    utils.delete_logger(name="resnet18_" + str(index), logger=logger)
    return tc_acc, vc_acc #ENV['best_acc']

if __name__ == "__main__":
    if args.dataset == "cifar10":
        config_path = "configs/config_10.json"
    elif args.dataset == "cifar100":
        config_path = "configs/config_100.json"
    elif args.dataset == "svhn":
        config_path = "configs/svhn.json"
    elif args.dataset == "purchase":
        config_path = "configs/purchase.json"
    elif args.dataset == "locations":
        config_path = "configs/locations.json"

    with open(config_path) as f:
        cfg = json.load(f)
    
    # main function
    if args.global_mode == "train":
        results = []
        for i in range(args.s_model, args.t_model):
            # try:
            tc_acc, vc_acc = main(cfg, aug_type = args.aug_type, index = i)
            print(tc_acc, vc_acc)
            # except:
            #     print("i:", i)
            #     results.append(i)
        print(results)
    elif args.global_mode == "decide":

        # decide the hyper-parameter
        results = []
        for i in range(len(cfg["augmentation_params"][args.aug_type])):
            vc_acc = main(cfg, aug_type = args.aug_type, index = 0, aug_index = i)
            results.append(vc_acc)
        print(results)
    
    elif args.global_mode == "gap":

    
        # Training and Test Gap
        print("need set the training set")
        exit(0)
        info = dict()
        info2 = dict()
        totestl = []
        it = [1,2,3,4,5,6,7,8,9,10,11,12]
        for i in it:
            print(cfg["training_augmentations"][i])
            tc_acc_list = []
            vc_acc_list = []
            for  j in range(10, 20):
                print(j, ", ok")
                tc_acc, vc_acc = main(cfg, aug_type = cfg["training_augmentations"][i], index = j, aug_index = 0)
                vc_acc_list.append(vc_acc)
                tc_acc_list.append(tc_acc)
            print(np.mean(tc_acc_list))
            print(np.std(tc_acc_list))
            print(np.mean(vc_acc_list))
            print(np.std(vc_acc_list))
            info2[cfg["training_augmentations"][i]] = (np.mean(tc_acc_list), np.std(tc_acc_list), np.mean(vc_acc_list), np.std(vc_acc_list))
            info[cfg["training_augmentations"][i]] = (tc_acc_list, vc_acc_list)
        # f = open(args.dataset + "_gap.txt", "w")
        # f.write(str(info))
        # f.close()
        # f = open(args.dataset + "_gap_info.txt", "w")
        # f.write(str(info2))
        # f.close()
    elif args.global_mode == "adv":
        tc_acc_list = []
        vc_acc_list = []
        for  j in range(10, 13):
            print(j, ", ok")
            tc_acc, vc_acc = main(cfg, aug_type = args.aug_type, index = j, aug_index = 0)
            vc_acc_list.append(vc_acc)
            tc_acc_list.append(tc_acc)
        print(np.mean(vc_acc_list))
        print(np.std(vc_acc_list))
    
    