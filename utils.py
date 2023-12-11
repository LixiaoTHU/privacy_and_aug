from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_path(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return path

def load_model(model_path, model, device='cpu'):
    checkpoint = torch.load(model_path+'.dat', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    return checkpoint

def save_model(model_path, model, epoch, save_best=False):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    with open(model_path+'.dat', 'wb') as f:
        torch.save(state, f)
    if save_best:
        with open(model_path + '_best' + '.dat', 'wb') as f:
            torch.save(state, f)

def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              '\tglobal_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = '\t' + key + '=' + value
        else:
            display += '\t' + str(key) + '=%.4f' % value
    display += '\ttime=%.2fit/s' % (1. / time_elapse)
    return display

    
def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def delete_logger(name, logger):
    for handler in logger.handlers:
        handler.close()

    logger.handlers = []

    logging.Logger.manager.loggerDict.pop(name, None)
    logging.shutdown()