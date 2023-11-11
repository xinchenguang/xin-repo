from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import pandas as pd
import glob
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
import os
from torch.optim import Adam
from PIL import Image, ImageOps
import torchvision
import torch
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
#张量化和归一化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
#这应该是获取mnist数据集
data_train = datasets.ImageFolder(root = "E:\历史代码测试保存\报告\软件工程\malimg_dataset\malimg_dataset\malimg_paper_dataset_imgs",
                            transform=transform)
print(data_train.classes)
print(data_train[0])
data_test = datasets.ImageFolder(root="E:/历史代码测试保存/报告/软件工程/malimg_dataset/malimg_dataset/test",transform = transform)

#数据处理器，批量处理，转换格式
data_loader_train = DataLoader(dataset=data_train,batch_size = 64,shuffle = True)

data_loader_test = DataLoader(dataset=data_test,batch_size = 64,shuffle = True)

import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        print(x.shape)
        x = x.view(-1, 128 * 28*28)
        print(x.shape)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = CNN(26)
#Adam是优化器，设置其学习率为0.001，使得模型能够更好地拟合数据
optimizer = Adam(model.parameters(),lr = 0.001)
learning_rate = 0.001
criterion = nn.CrossEntropyLoss() # 交叉熵损失

epoch = 1
def train():
    for idx,(input,target)  in enumerate(data_loader_train):#idx表示data_loader中的第几个数据，元组是data_loader的数据
        optimizer.zero_grad()#将梯度置0，这样step只会更新当前的梯度，不会更新之前的
        output = model(input)#调用模型，得到预测值
        loss = criterion(output,target)#调用损失函数，得到损失,是一个tensor,值越小代表预测越准确
        loss.backward()#反向传播
        optimizer.step()#梯度的更新
        if idx % 50 == 0:
            print(epoch,idx,loss.item())
            #print('input=',input)
    #保存模型以及优化器
    torch.save(model.state_dict(),'./m_net.pth')#保存模型参数，state_dict用来获取数据，save用来保存数据
    torch.save(optimizer.state_dict(),"./m_optimizer.pth")#保存优化器
#train()
model.load_state_dict(torch.load("./m_net.pth"))
optimizer.load_state_dict(torch.load("./m_optimizer.pth"))

def test():
    loss_list = []
    acc_list = []
    for idx,(input,target) in enumerate(data_loader_test):
        with torch.no_grad():#不计算梯度
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss)
            #计算准确率，output大小[batch_size,10] target[batch_size] batch_size是多少组数据，10列是每个数字概率
            pred = output.max(dim = -1)[-1]#获取最大值位置
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率：",np.mean(acc_list),"平均损失：",np.mean(loss_list))

def image_loader():
    image = Image.open('E:/历史代码测试保存/报告/软件工程/malimg_dataset/malimg_dataset/malimg_paper_dataset_imgs/Instantaccess/00ceb3faed1b195c080d6439f61071c5.png')
    img = transform(image)
    #image = image.point(lambda x:0 if x<80 else 255, '1')
    #img = np.array(image).astype(np.float32)
    #img = torch.from_numpy(img)
    img = img.to('cpu')
    print(img)
    output = model(img)  # 进行推理，output为模型的输出结果
    print('output',output)
    prob = F.softmax(output, dim=1)  # 对输出结果进行softmax处理，得到概率值
    prob = prob.detach().numpy()  # 将概率值转换为numpy数组，以进行后续处理
    result = np.argmax(prob)  # 将概率值最大的类别索引作为预测结果
    print('Predicted digit:', result)  # 输出预测结果
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title('my image :' + str(result))
    # 显示图像
    plt.show()

#test()
#image_loader()
