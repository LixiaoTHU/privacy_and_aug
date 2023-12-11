import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10, width_num = 16, channels = 3, size = 32):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.width_num = width_num
        self.channels = channels
        self.size = size
        self._build_model()
        
    
            
    def _build_model(self):

        self.conv0 = nn.Conv2d(self.channels, 16*self.width_num, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv1 = nn.Conv2d(16*self.width_num, 32*self.width_num, kernel_size=2, stride=2, padding=0, bias=False)

        self.conv2 = nn.Conv2d(32*self.width_num, 32*self.width_num, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv3 = nn.Conv2d(32*self.width_num, 32*self.width_num, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv4 = nn.Conv2d(32*self.width_num, 32*self.width_num, kernel_size=2, stride=2, padding=0, bias=False)

        self.conv5 = nn.Conv2d(32*self.width_num, 32*self.width_num, kernel_size=3, stride=1, padding=0, bias=False)
        
        self.fc1 = nn.Linear(32*self.width_num, 200, bias=False)
        self.bng = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, self.num_classes)

        self.batch_norm_0 = nn.BatchNorm2d(16*self.width_num)
        self.batch_norm_1 = nn.BatchNorm2d(32*self.width_num)
        self.batch_norm_2 = nn.BatchNorm2d(32*self.width_num)
        self.batch_norm_3 = nn.BatchNorm2d(32*self.width_num)
        self.batch_norm_4 = nn.BatchNorm2d(32*self.width_num)
        self.batch_norm_5 = nn.BatchNorm2d(32*self.width_num)
        
    def base_forward(self, x):
        x = F.relu(self.batch_norm_0(self.conv0(x)))
        x = F.relu(self.batch_norm_1(self.conv1(x)))
        x = F.relu(self.batch_norm_2(self.conv2(x)))
        x = F.relu(self.batch_norm_3(self.conv3(x)))
        x = F.relu(self.batch_norm_4(self.conv4(x)))
        x = F.relu(self.batch_norm_5(self.conv5(x)))
        x = F.avg_pool2d(x, self.size//4-2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bng(self.fc1(x)))
        x = self.fc2(x)
        
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
