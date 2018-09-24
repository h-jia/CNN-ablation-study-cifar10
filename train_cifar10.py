import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pickle as pkl

import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns

from model.models import *

transformer = transforms.Compose([       
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transformer)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transformer)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(epoch):
    train_loss = []
    acc = []
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):        
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        _, predicts = outputs.max(1)
        acc.append(torch.eq(predicts, targets).sum().item() / predicts.size()[0])
        
        
    
    loss = sum(train_loss) / len(train_loss)
    accuracy = sum(acc) / len(acc) * 100
    msg = 'Epoch:{:d} train loss: {:3f}  accuracy: {:3f}% \n'.format(epoch, loss, accuracy)
    print(msg)
    with open('log/record_v2.txt', 'a') as file:
        file.write(msg)
    return loss, accuracy
    
def test(epoch):
    test_loss = []
    acc = []    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()        
            outputs = model(inputs)
            loss = criterion(outputs, targets)            

            test_loss.append(loss.item())
            _, predicts = outputs.max(1)
            acc.append(torch.eq(predicts, targets).sum().item() / predicts.size()[0])

        loss = sum(test_loss) / len(test_loss)
        accuracy = sum(acc) / len(acc) * 100
    msg = 'Epoch:{:d} test loss: {:3f}  accuracy: {:3f}% \n'.format(epoch, loss, accuracy)
    print(msg)
    with open('log/record_v2.txt', 'a') as file:
        file.write(msg)
    return loss, accuracy

#VGG(), VGG_wo_bn(), VGG_wo_maxpool(), VGG_wo_relu(), VGG_wo_bn_relu(),     
#          VGG_linear_1(), VGG_linear_2(), VGG_linear_3(), VGG_linear_4(), VGG_linear_5(),
#          VGG_linear_6()

models = [VGG_linear_7(), VGG_shallow_1(), VGG_shallow_2(), VGG_bottleneck_1(), 
          VGG_bottleneck_2(), VGG_sigmoid()]


model_log = []
for idx, model in enumerate(models):
    model = model.cuda()
    model = nn.DataParallel(model)    
    log = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(100):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        log.append([train_loss, train_acc, test_loss, test_acc])
    model_log.append(log)
    
with open('log/model_log.pkl', 'wb') as file:
    pkl.dump(model_log, file)