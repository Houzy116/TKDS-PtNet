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
all=[]
for i,v in enumerate(loaders['train']):
    all.append(v[0])
for i,v in enumerate(loaders['val']):
    all.append(v[0])
# %%
import torch
a=torch.cat(all,axis=0)
a.shape
# %%
a=a.view([1049033,6,-1])
# %%
a.shape
# %%
a=np.array(a)
# %%
a=a.transpose(1,0,2)
# %%
a=a.reshape(6,-1)
# %%
a.shape
# %%
def get_meanandstd(a):
    mean=[]
    std=[]
    for i in range(a.shape[0]):
        m=np.mean(a[i,:])
        mean.append(m)
        s=np.std(a[i,:],ddof=1)
        std.append(s)
    return mean,std

# %%
get_meanandstd(a)
# %%
all[0].shape
# %%
import matplotlib.pyplot as plt
plt.imshow(all[0][0,0,:,:])
plt.show()
plt.imshow(all[0][0,1,:,:])
plt.show()
plt.imshow(all[0][0,3,:,:])
plt.show()
# %%
len(all)
# %%
