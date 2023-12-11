import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

criterion_kl = nn.KLDivLoss(reduction='sum')

def cal_adv(model, criterion, aug_type, imgs, cids, eps = 8):
    steps = 3
    step_size = 1.25 * eps / steps
    # steps = 20
    # step_size = 1
    model.eval()
    x_natural = imgs
    x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-eps / 255, eps / 255)
    if aug_type == "trades" or aug_type == "TradesAWP":
        logits = model.base_forward(x_natural)
        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(model(x_adv), F.softmax(logits, dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + (step_size / 255) * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - eps / 255), x_natural + eps / 255)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv
    elif aug_type == "pgdat" or aug_type == "AWP":
        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                pred = model(x_adv)
                loss = criterion(pred, cids)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + (step_size / 255) * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - eps / 255), x_natural + eps / 255)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv