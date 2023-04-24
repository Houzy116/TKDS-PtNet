#%%
import torch.nn as nn
import torch
import torch.nn.functional as F
class sentinel2net6(nn.Module):
    def __init__(self,n_class,in_planes10,in_planes20):
        super().__init__()
        self.n_class=n_class
        self.in_planes10=in_planes10
        self.in_planes20=in_planes20
        self.conv_10=nn.Conv2d(self.in_planes10,64,kernel_size=1,stride=1,bias=False)
        self.conv_20=nn.Conv2d(self.in_planes20,64,kernel_size=1,stride=1,bias=False)
        self.layer10=nn.Sequential(SEBlock(64,64,False),SEBlock(64,128,True))
        self.layer20=nn.Sequential(SEBlock(64,64,False),SEBlock(64,128,False))
        self.layer_all=nn.Sequential(SEBlock(256,256,False),SEBlock(256,512,False))
        self.layer_diff=nn.Sequential(SEBlock(512,512,False),SEBlock(512,1024,False))
        self.layer_mlp=Mlp(1024,self.n_class)

    def forward(self,x):
        img10=x[0].to(torch.float32)
        img20=x[1].to(torch.float32)
        if img10.shape[1]%2!=0 or img20.shape[1]%2!=0:
            raise('channels length should be an even number')
        channels_len_10=int(img10.shape[1]/2)
        channels_len_20=int(img20.shape[1]/2)
        pre10,pre20=img10[:,:channels_len_10,:,:],img20[:,:channels_len_20,:,:]
        post10,post20=img10[:,channels_len_10:,:,:],img20[:,channels_len_20:,:,:]
        pre10,post10=self.conv_10(pre10),self.conv_10(post10)#>>n,64,6,6
        pre20,post20=self.conv_20(pre20),self.conv_20(post20)#>>n,64,3,3
        # pre10=self.layer10(pre10)
        pre10,post10=self.layer10(pre10),self.layer10(post10)#全卷积+通道注意力+patch编码  6*6>>>3*3
        pre20,post20=self.layer20(pre20),self.layer20(post20)#全卷积+通道注意力  3*3>>>3*3
        pre=torch.cat((pre10, pre20), 1)
        post=torch.cat((post10, post20), 1)
        pre,post=self.layer_all(pre),self.layer_all(post)#全卷积+通道注意力 3*3>>>3*3
        diff=post.sub(pre)
        diff=self.layer_diff(diff)# 3*3>>>1*1
        out=F.avg_pool2d(diff,diff.size(2))
        out = out.view(out.size(0), -1)
        out = self.layer_mlp(out)
        return out

class SEBlock(nn.Module):
    def __init__(self,in_planes,planes,patchembed=True,act_layer=nn.GELU):
        '''
        in_planes:输入通道
        planes:输出通道
        patchembed:size是否减半
        '''
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=1,stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        if patchembed:
            self.conv3=nn.Conv2d(planes,planes,kernel_size=2,stride=2, bias=False)
            self.shortcut=nn.AvgPool2d(kernel_size=2,stride=2)
        else:
            self.conv3=nn.Conv2d(planes,planes,kernel_size=1,stride=1, bias=False)               
        self.fc1 = nn.Conv2d(in_planes, in_planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(in_planes//16, in_planes, kernel_size=1)
        self.fc3 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc4 = nn.Conv2d(planes//16, planes, kernel_size=1)
        self.act=act_layer()
    def forward(self, x):
        out = self.act(self.bn1(x))

        w1 = F.avg_pool2d(out, out.size(2))
        w1 = self.act(self.fc1(w1))
        w1 = F.sigmoid(self.fc2(w1))

        out = out * w1

        out = self.conv1(out)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else out
        out = self.conv2(self.act(self.bn2(out)))
        out = self.conv3(self.act(self.bn3(out)))

        w2 = F.avg_pool2d(out, out.size(2))
        w2 = self.act(self.fc3(w2))
        w2 = F.sigmoid(self.fc4(w2))

        out = out * w2

        out += shortcut
        # print(out.size())
        return out

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features,n_class,act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, int(in_features/4))
        self.act = act_layer()
        self.fc2 = nn.Linear(int(in_features/4), int(in_features/16))
        self.fc3 = nn.Linear(int(in_features/16), n_class)
        self.drop = nn.Dropout(drop)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x
# #%%%
# import sys
# sys.path.append('..')
# import loader
# loaders=loader.get_loader()
# train_loader=loaders['train']
# for id,batch in enumerate(train_loader,0):
#     break
# # %%
# img10=batch[0]
# img20=batch[1]
# print(img10.shape)
# print(img20.shape)
# # %%
# img10[:,:4,:,:].shape

# # %%
# img10[:,4:,:,:].shape
# # %%
# model=sentinel2net6(2,4,6)
# # %%
# pred=model([img10,img20])
# gt=batch[2].long()
# # %%
# loss_fun=F.cross_entropy
# # %%
# loss_fun(pred,gt)
# # %%
# pred=torch.argmax(pred,dim=1)
# # %%
# print(pred)
# # %%
# print(gt)
# # %%
# from run.metric_tool import ConfuseMatrixMeter
# # %%
# metric=ConfuseMatrixMeter(2)
# # %%
# metric.update_cm(pr=pred.cpu().numpy(),gt=gt.cpu().numpy())
# # %%
