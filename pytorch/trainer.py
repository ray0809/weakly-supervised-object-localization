import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from model import ResNet50Net
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_cifar10_dataloader():

    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=True,
                                                transform=transforms.ToTensor())

    valid_dataset = torchvision.datasets.CIFAR10(root='data/',
                                                train=False,
                                                transform=transforms.ToTensor())


    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                            shuffle=True, num_workers=8)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64,
                                            shuffle=False, num_workers=8)


    return trainloader, validloader



def train():
    
    # init net
    net = ResNet50Net(224, 10)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    net = net.to(device)

    # load dataset
    trainloader, validloader = create_cifar10_dataloader()


    # start training
    if not os.path.isdir('./checkpoint'):
        os.makedirs('./checkpoint')
    epoches = 10
    best_acc = 0.0
    for i in range(epoches):
        net.train()

        train_correct = 0
        train_total = 0
        for (imgs, labels) in tqdm(trainloader):
            imgs = imgs.to(device)
            labels = labels.to(device).type(torch.cuda.LongTensor)

            output = net(imgs)
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.shape[0]
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('############## taining ##############')
        print('{}/{}, loss:{}, training Accuracy:{}'.format(i+1, epoches, loss.item(), train_correct / train_total))
        print('############## taining ##############')
    

        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for imgs, labels in tqdm(validloader):
                imgs = imgs.to(device)
                labels = labels.to(device).type(torch.cuda.LongTensor)
                output = net(imgs)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
            valid_acc = correct / total
            
            print('############## testing ##############')
            print('testing Average Accuracy is {}'.format(valid_acc))  
            print('############## testing ##############')

            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(net.state_dict(), './checkpoint/resnet50.pkl')

if __name__ == "__main__":
    train()


    




