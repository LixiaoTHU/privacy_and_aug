import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_classes=10, size = 28*28):
        super(MLP, self).__init__()

        self.size = size
        self.num_classes = num_classes
        self.batch_norm_0 = nn.BatchNorm1d(size)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.batch_norm_3 = nn.BatchNorm1d(128)
        self.batch_norm_4 = nn.BatchNorm1d(128)

        self.fc0 = nn.Linear(size, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, num_classes)


    def base_forward(self, x):
        x = x.view(-1, self.size)

        x = F.relu(self.batch_norm_1(self.fc0(x)))

        x = F.relu(self.batch_norm_2(self.fc1(x)))

        x = F.relu(self.batch_norm_3(self.fc2(x)))

        x = F.relu(self.batch_norm_4(self.fc3(x)))

        x = self.fc4(x)
        return x

    def forward(self, x, require_logits = False):
        logits = self.base_forward(x)
        if require_logits:
            return F.log_softmax(logits, dim = 1), logits
        return F.log_softmax(logits, dim = 1)

    def forward_w_temperature(self, x, T=1):
        logits = self.base_forward(x)
        scaled_logits = logits/T
        return F.softmax(scaled_logits, dim=1)

