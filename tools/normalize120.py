#%%
import sys 
sys.path.append('..')
import loader
import run
import warnings
import numpy as np
warnings.filterwarnings("ignore")

loaders=loader.get_loader()
# %%
sum=[0]*6
st_sum=[0]*6
n=0
dataset_l=len(loaders['train'])
for i,v in enumerate(loaders['train']):
    L=v[0].shape[0]
    for l in range(L):
        img=v[0][l,:,:,:]
        n+=120*120
        for b in range(6):
            sum[b]+=img[b,:,:].sum()
    sys.stdout.write('\r'+str(i)+'/'+str(dataset_l))
    sys.stdout.flush()
mean=sum/np.array(n)
print('mean:',mean)


for i,v in enumerate(loaders['train']):
    L=v[0].shape[0]
    for l in range(L):
        img=v[0][l,:,:,:]
        n+=120*120
        for b in range(6):
            array=np.array(img[b,:,:]-mean[b])
            st_sum[b]+=np.sum(array**2)
    sys.stdout.write('\r'+str(i)+'/'+str(dataset_l))
    sys.stdout.flush()
st=np.sqrt(np.array(st_sum)/(n-1))
print('std:',st)
# %%
import torch
imgs=torch.load('/ssd/hk/Syria_samples/split_havedamaged_size120_all_v2/val.pth')
import imageio
i1=imageio.imread(imgs[0][0])
i2=imageio.imread(imgs[1][0])
# %%
import random
hv_flip_together(i1, i2)
# %%
def hv_flip_together(image, image2):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        image = np.flip(image,axis=0)
        image2 = np.flip(image2,axis=0)
    if random.random() > 0.5:
        image = np.flip(image,axis=1)
        image2 = np.flip(image2,axis=1)
    # image = tf.to_tensor(image)
    # image2 = tf.to_tensor(image2)
    return image, image2
# %%
torch.tensor(i1)
# %%
from torchvision import utils,transforms
a=transforms.ToTensor()
a(i1)
# %%
