#%%
from docutils import TransformSpec
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class sentinel2net6_trf(nn.Module):
    def __init__(self,n_class,in_planes10,in_planes20):
        super().__init__()
        self.n_class=n_class
        self.in_planes10=in_planes10
        self.in_planes20=in_planes20
        self.fn_10=nn.Conv2d(self.in_planes10,64,kernel_size=1,stride=1,bias=False)
        self.fn_20=nn.Conv2d(self.in_planes20,64,kernel_size=1,stride=1,bias=False)
        self.layer10=nn.Sequential(MyBlock(64,64,6,[4,8],False),MyBlock(64,128,6,[4,4],True))
        self.layer20=nn.Sequential(MyBlock(64,64,3,[4,8],False),MyBlock(64,128,3,[4,4],False))
        self.layer_all=nn.Sequential(MyBlock(256,256,3,[4,16],False),MyBlock(256,512,3,[8,8],False))
        self.layer_diff=nn.Sequential(MyBlock(512,512,3,[16,8],False),MyBlock(512,1024,3,[16,16],False))
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
        pre10,post10=self.fn_10(pre10),self.fn_10(post10)#>>n,64,6,6
        pre20,post20=self.fn_20(pre20),self.fn_20(post20)#>>n,64,3,3
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

class MyBlock(nn.Module):
    def __init__(self,in_planes,planes,size,head_num,patchembed=True,act_layer=nn.GELU):
        '''
        in_planes:输入通道
        planes:输出通道
        patchembed:size是否减半
        '''
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.att1=Attention(in_planes,size,head_num[0])
        self.bn3 = nn.BatchNorm2d(planes)
        self.att2=Attention(planes,size,head_num[1])
        self.bn4 = nn.BatchNorm2d(planes)
        if patchembed:
            self.conv=nn.Conv2d(planes,planes,kernel_size=2,stride=2, bias=False)
            self.shortcut=nn.AvgPool2d(kernel_size=2,stride=2)
        self.fc = nn.Conv2d(in_planes, planes, kernel_size=1)
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

        out = self.att1(self.act(self.bn2(out)))
        out = self.act(self.fc(out))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else out
        out = self.att2(self.act(self.bn3(out)))

        if hasattr(self, 'shortcut'):
            out = self.conv(self.act(self.bn4(out)))

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

class Attention(nn.Module):

    def __init__(self, dim, size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.size = size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * size - 1) * (2 * size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.size)
        coords_w = torch.arange(self.size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.size - 1
        relative_coords[:, :, 0] *= 2 * self.size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        H,W,C,B_=x.shape[2],x.shape[3],x.shape[1],x.shape[0]
        N=W*H
        x=x.permute(0,2,3,1).contiguous().view(-1,N,C)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.size * self.size, self.size * self.size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x=x.permute(0,2,1).contiguous().view(B_,C,H,W)
        return x
# #%%
# a=torch.rand(512,64,6,6)
# att=Attention(64,6,4)
# #%%
# block=MyBlock(64,128,6,[4,4],True)
# block(a).shape
# #%%
# model=sentinel2net6_trf(2,4,6)
# model([torch.rand(512,8,6,6),torch.rand(512,12,3,3)])
# #%%
# # #%%%
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

# %%
