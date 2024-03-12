#%%
import torch.nn as nn
import torch
import math
from einops import rearrange


class ptnetsize120(nn.Module):
    """
    block: A sub module
    """

    def __init__(self, layers=[2, 2, 1, 1], n_class=2, model_path=None):
        super(ptnetsize120, self).__init__()
        self.inplanes = 64
        self.modelPath = model_path
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.make_stack(64, layers[0], stride=2)
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        self.avgpool = nn.AvgPool2d(5, stride=5)
        self.se=SELayer(512)
        self.fn=nn.Conv2d(512, 128, kernel_size=1, stride=1,
                        bias=False)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 128, 36))
        # self.token=Semantic_tokens(channel=128,token_len=32)
        self.trf1=Transformer(128,1,8,128,128,0)
        self.trf2=Transformer(128,1,8,128,128,0)

        self.fn1=nn.Linear(128,4)
        self.ln=nn.LayerNorm(144)
        self.fn2=nn.Linear(144,n_class)
        self.relu=nn.ReLU(inplace=True)

        # initialize parameters
        # self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.stack1(x)
        x = self.stack2(x)
        x=self.avgpool(x)
        x = self.se(x)
        x=self.fn(x)
        B,C,W,H=x.shape
        x = x.view(B,C,-1).contiguous()
        x=self.relu(x)
        x=rearrange(x, 'b c l -> b l c')
        # print(x.shape)
        x=self.trf1(x)
        x=self.trf2(x)
        # x=self.trf3(x)
        x=self.fn1(x)
        x=self.relu(x)
        x=x.view(B,-1)
        x=self.ln(x)
        x=self.fn2(x)
        x=self.relu(x)
        return x



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Semantic_tokens(nn.Module):
    def __init__(self,channel=64,token_len=16):
        super(Semantic_tokens,self).__init__()
        self.conv=nn.Conv2d(channel, token_len, kernel_size=1,padding=0, bias=False)

    def forward(self, x):
        r=x
        x=x.unsqueeze(dim=-1).contiguous()
        x=self.conv(x)
        x=x.squeeze(dim=-1).contiguous()
        x = torch.softmax(x, dim=-1)
        tokens = torch.einsum('bln,bcn->blc', x, r)

        return tokens
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, out_dim, dropout = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
        #32,1,8,64,64

class Residual(nn.Module):
    def __init__(self, input_dim,fn,downsample=False):
        super().__init__()
        self.fn = fn
        if downsample:
            self.downsample=nn.Sequential(
                nn.Linear(input_dim,input_dim*2),nn.LayerNorm(input_dim*2))
        else:
            self.downsample=None
    def forward(self, x, **kwargs):
        if self.downsample is None:
            return self.fn(x, **kwargs) + x
        else:
            x_re=self.downsample(x)

            return self.fn(x, **kwargs) + x_re

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, out_dim, dropout):
        super().__init__()
        if out_dim!=dim:
            self.downsample=True
        else:
            self.downsample=False
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(input_dim=dim,fn=PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(input_dim=dim,fn=PreNorm(dim, FeedForward(dim, out_dim, dropout = dropout)),downsample=self.downsample)
            ]))
    def forward(self, x):
        for attn,ff in self.layers:
            # print(1)
            x = attn(x)
            # break
            x = ff(x)
        return x


