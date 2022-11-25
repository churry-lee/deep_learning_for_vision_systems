#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2 as cv
import pandas as pd
import time

from typing import List, Dict, Tuple
from models.myNet import Net
from models.AlexNet import AlexNet


def data_loader():
    pre_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=pre_processing)
    test_data = datasets.CIFAR10(root='../data', train=False, download=True, transform=pre_processing)

    train_len = int(len(data_set) * 0.85)
    valid_len = int(len(data_set) * 0.15)
    train_data, valid_data = torch.utils.data.random_split(data_set, [train_len, valid_len])


    return (train_data, valid_data, test_data)

def model_loader():
    net = Net().to(device)

    return net


def get_accuracy(y, label):
    y_idx = torch.argmax(y, dim=1)
    result = y_idx - label

    num_correct = 0
    for i in range(len(result)):
        if result[i] == 0:
            num_correct += 1

    return num_correct/y.shape[0]


def train(dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    train_loss_list, train_acc_list = [], []

    start_time = time.time()
    for batch, (x, y) in enumerate(dataloader):
        model.train()
        # x: 입력, y: 정답(레이블)을 받아온 후 device에 올려줌
        x, y = x.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(x)
        loss = loss_fn(pred, y)  # 손실함수 계산

        # 역전파
        optimizer.zero_grad() # 학습 수행 전 미분값을 0으로 초기화(학습전 반드시 처리 필요)
        loss.backward()       # 가중치와 편향에 대한 기울기 계산
        optimizer.step()      # 가중치와 편향 업데이트

        # 학습 정확도 및 손실함수 값 기록
        train_acc = get_accuracy(pred, y)  # 정확도 계산

        train_loss_list.append(loss.item())
        train_acc_list.append(train_acc)

        if (batch+1) % num_batches == 0:
            print(f'step: {batch+1}/{num_batches} | {time.time() - start_time:.2f} s/step | ', end='')
            print(f'train loss: {np.mean(train_loss_list):.4f} | train acc: {np.mean(train_acc_list):.4f} | ', end='')

    return np.mean(train_loss_list), np.mean(train_acc_list)


def valid(dataloader, model, loss_fn):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        val_loss_list, val_acc_list = [], []

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            correct = model(x)
            val_loss = loss_fn(correct, y)

            val_acc = get_accuracy(correct, y)

            val_loss_list.append(val_loss.item())
            val_acc_list.append(val_acc)

        print(f'valid loss: {np.mean(val_loss_list):.4f} | valid acc: {np.mean(val_acc_list):.4f}')
        return np.mean(val_loss_list), np.mean(val_acc_list)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


def main():

    net = model_loader()
    print(net)

    (train_data, valid_data, test_data) = data_loader()
    print(len(train_data), len(valid_data), len(test_data))

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()  # 손실함수 설정
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # 최적화 설정

    total_train_loss, total_train_acc = [], []
    total_val_loss, total_val_acc = [], []

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        mean_train_loss, mean_train_acc = train(train_loader, net, loss_fn, optimizer)
        mean_val_loss, mean_val_acc = valid(valid_loader, net, loss_fn)

        result = pd.DataFrame({
            "train_loss": mean_train_loss,
            "train_acc" : mean_train_acc,
            "val_loss"  : mean_val_loss,
            "val_acc"   : mean_val_acc
        }, index=["epoch"])
        result.to_csv("./test_save", mode="a", sep=",", na_rep="NaN", float_format="%.4f", index=False)

    # plt.figure(figsize=(10, 5))
    # plt.title("Training and Validation Loss")
    # plt.plot(total_train_loss, label="train")
    # plt.plot(total_val_loss, label="val")
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.show()

    print("Done!")


if __name__ == "__main__":
    # device = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using *{}* device".format(device))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    main()
