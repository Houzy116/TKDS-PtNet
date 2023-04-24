#%%
from docutils import TransformSpec
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_




class simplenet(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        self.conv=nn.Conv2d(4,16,kernel_size=2,stride=2,bias=False)
        self.bn1=nn.BatchNorm2d(44)
        self.fn1=nn.Conv2d(44,64,kernel_size=1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(64)
        self.fn2=nn.Conv2d(64,128,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(128)
        self.fn3=nn.Conv2d(128,256,kernel_size=1,stride=1,bias=False)
        self.bn4=nn.BatchNorm2d(256)
        self.fn4=nn.Conv2d(256,512,kernel_size=1,stride=1,bias=False)
        self.conv2=nn.Conv2d(512,512,kernel_size=3,stride=3,bias=False)
        self.act=nn.GELU()
        self.mlp=Mlp(512,n_class)
    def forward(self,x):
        img10=x[0].to(torch.float32)
        img20=x[1].to(torch.float32)
        if img10.shape[1]%2!=0 or img20.shape[1]%2!=0:
            raise('channels length should be an even number')
        channels_len_10=int(img10.shape[1]/2)
        channels_len_20=int(img20.shape[1]/2)
        pre10,pre20=img10[:,:channels_len_10,:,:],img20[:,:channels_len_20,:,:]
        post10,post20=img10[:,channels_len_10:,:,:],img20[:,channels_len_20:,:,:]
        pre10=self.conv(pre10)
        post10=self.conv(post10)
        pre=torch.cat((pre10,pre20),1)
        post=torch.cat((post10,post20),1)
        input=torch.cat((pre,post),1)
        input=self.fn1(self.act(self.bn1(input)))
        input=self.fn2(self.act(self.bn2(input)))
        input=self.fn3(self.act(self.bn3(input)))
        input=self.fn4(self.act(self.bn4(input)))
        input=self.conv2(self.act(input))
        input = input.view(input.size(0), -1)
        input=self.mlp(input)
        return input

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features,n_class,act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, int(in_features/4))
        self.act = act_layer()
        self.fc2 = nn.Linear(int(in_features/4), int(in_features/16))
        self.fc3 = nn.Linear(int(in_features/16), n_class)
        self.drop = nn.Dropout(drop)
        self.ln1=nn.LayerNorm(int(in_features/4))
        self.ln2=nn.LayerNorm(int(in_features/16))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(self.ln1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(self.ln2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x

