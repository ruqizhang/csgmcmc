'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable
import numpy as np
import random

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Ensemble')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
net = ResNet18()
if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True

criterion = nn.CrossEntropyLoss()

def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
             right += 1.0
    return right/len(truth)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_list = []
    truth_res = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            truth_res += list(targets.data)
            outputs = net(inputs)
            pred_list.append(F.softmax(outputs,dim=1))
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/len(testloader), correct, total,
        100. * correct.item() / total))
    pred_list = torch.cat(pred_list,0)
    return pred_list,truth_res

pred_list = []
num_model = 12
for m in range(num_model):
    net.load_state_dict(torch.load(args.dir + '/cifar_model_%i.pt'%(m)))
    pred, truth_res = test()
    pred_list.append(pred)

fake = sum(pred_list)/num_model
values, pred_label = torch.max(fake,dim = 1)
pred_res = list(pred_label.data)
acc = get_accuracy(truth_res, pred_res)
print(acc)
