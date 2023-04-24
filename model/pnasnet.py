#%%
import torch.nn as nn
import torch
import math
class pnasnet(nn.Module):
    def __init__(self,n_class=2):
        super(pnasnet,self).__init__()

        self.conv1=nn.Conv2d(6,128,kernel_size=9,padding=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.mp1=nn.MaxPool2d(8,stride=8)
        self.dropout1=nn.Dropout(0.255)

        self.conv2=nn.Conv2d(128,256,kernel_size=9,padding=4)
        # self.bn2 = nn.BatchNorm2d(64)
        self.mp2=nn.MaxPool2d(8,stride=8,padding=1)
        self.dropout2=nn.Dropout(0.225)

        # self.conv3=nn.Conv2d(64,128,kernel_size=7)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.mp3=nn.MaxPool2d(6,stride=1)
        # self.dropout3=nn.Dropout(0.5)

        self.fc1=nn.Linear(1024,128)
        self.n1 = nn.LayerNorm(128)
        self.fc2=nn.Linear(128,n_class)
    def forward(self,x):

        x=self.conv1(x)
        # x=self.bn1(x)
        # print(x.shape)
        x=self.relu(x)
        x=self.mp1(x)
        # print(x.shape)
        x=self.dropout1(x)


        x=self.conv2(x)
        # # x=self.bn2(x)
        # # print(x.shape)
        x=self.relu(x)
        x=self.mp2(x)
        # # print(x.shape)
        x=self.dropout2(x)


        # x=self.conv3(x)
        # x=self.bn3(x)
        # # print(x.shape)
        # x=self.relu(x)
        # x=self.mp3(x)
        # # print(x.shape)
        # x=self.dropout3(x)
        # # print(x.shape)
        
        x = x.view(x.size(0), -1)

        x=self.fc1(x)
        x=self.relu(x)
        # # print(x.shape)
        x=self.n1(x)
        x=self.fc2(x)
        # x=self.relu(x)
        return x

# %%
# 222222222222


# %%
