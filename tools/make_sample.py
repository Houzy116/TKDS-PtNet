#%%
import gdal
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from glob import glob
import shapely
# %%
def read_img(path):
    ds=gdal.Open(path)
    trf=ds.GetGeoTransform()
    img=ds.ReadAsArray()
    return {'trf':trf,'img':img}
def get_patch(data,size):
    img=data['img']
    trf=data['trf']
    shape=img.shape
    img=np.pad(img,((0,0),(0,size-shape[1]%size),(0,size-shape[2]%size)),'constant',constant_values=np.nan)
    shape=img.shape
    img=img.reshape([shape[0],int(shape[1]/size),-1,shape[2]])
    shape=img.shape
    img=img.reshape([shape[0],shape[1],shape[2],int(shape[3]/size),-1])
    return {'img':img.transpose((0,1,3,2,4)),'trf':(trf[0]-trf[1]/2,trf[1]*size,trf[2],trf[3]-trf[5]/2,trf[4],trf[5]*size)}
# %%
pathes=['../data/img/Livoberezhyny/Livoberezhyny_band2348/median-2022-2-24-2022-5-7.tif',
        '../data/img/Livoberezhyny/Livoberezhyny_band2348/post-2022-5-10-2022-7-1.tif',
        '../data/img/Livoberezhyny/Livoberezhyny_band2348/pre-2021-6-1-2021-10-1.tif'
        ]
for i in range(len(pathes)):
    path=pathes[i]
    data=read_img(path)
    print(data['img'].shape)
    data=get_patch(data,2)
    print(data['img'].shape)
    print(data['trf'])
    data['readme']='trf分别为patch左上角坐标（不是左上角像元的中心坐标）和patch的长度'
    torch.save(data,path.replace('.tif','_size_2.pth.tar'))



    
pathes=['../data/img/Livoberezhyny/Livoberezhyny_bandother/median-2022-2-24-2022-5-7.tif',
        '../data/img/Livoberezhyny/Livoberezhyny_bandother/post-2022-5-10-2022-7-1.tif',
        '../data/img/Livoberezhyny/Livoberezhyny_bandother/pre-2021-6-1-2021-10-1.tif'
        ]
for i in range(len(pathes)):
    path=pathes[i]   
    data=read_img(path)
    print(data['img'].shape)
    data=get_patch(data,1)
    data['img']=data['img'][:,:,:-1,:,:]
    print(data['img'].shape)
    print(data['trf'])
    data['readme']='trf分别为patch左上角坐标（不是左上角像元的中心坐标）和patch的长度'
    torch.save(data,path.replace('.tif','_size_1.pth.tar'))
    
#%%
damage_L_gpd=gpd.read_file(point_file)
damage_L_gpd[damage_L_gpd['d_Main_Dam']=='Destroyed']['geometry']
point2=damage_L_gpd[damage_L_gpd['d_Main_Dam']=='Destroyed']
point=point2['geometry']



patch=torch.load(patch_files[0])
mask=get_dam_mask(point,patch)
plt.figure(figsize=(30,15))
plt.imshow(mask)
point.plot(figsize=(30,15))
#%%
size=2
patch_files=[f'../data/img/Livoberezhyny/Livoberezhyny_band2348/pre-2021-6-1-2021-10-1_size_{size}.pth.tar',
            f'../data/img/Livoberezhyny/Livoberezhyny_band2348/median-2022-2-24-2022-5-7_size_{size}.pth.tar',
            f'../data/img/Livoberezhyny/Livoberezhyny_band2348/post-2022-5-10-2022-7-1_size_{size}.pth.tar',
            f'../data/img/Livoberezhyny/Livoberezhyny_bandother/pre-2021-6-1-2021-10-1_size_{int(size/2)}.pth.tar',
            f'../data/img/Livoberezhyny/Livoberezhyny_bandother/median-2022-2-24-2022-5-7_size_{int(size/2)}.pth.tar',
            f'../data/img/Livoberezhyny/Livoberezhyny_bandother/post-2022-5-10-2022-7-1_size_{int(size/2)}.pth.tar'
        ]
point_file=glob('../data/UNOSAT/LivoberezhnyiandZhovtnevyi_0507-08-12/*DA*.shp')[0]
def get_dam_mask(point,patch):
    trf=patch['trf']
    p_coor=[shapely.geometry.mapping(list(point)[i])['coordinates'] for i in range(len(point))]
    mask=np.zeros((patch['img'].shape[1],patch['img'].shape[2]))
    p_x_y=[[int((i[0]-trf[0])/trf[1]),int((i[1]-trf[3])/trf[5])] for i in p_coor]
    for i in p_x_y:
        mask[i[1],i[0]]=1
    return mask
def get_sample(point_file,patch_files):
    patch=torch.load(patch_files[0])
    damage_L_gpd=gpd.read_file(point_file)
    damage_L_gpd[damage_L_gpd['d_Main_Dam']=='Destroyed']['geometry']
    point=damage_L_gpd[damage_L_gpd['d_Main_Dam']=='Destroyed']['geometry']
    mask=get_dam_mask(point,patch)
    all_patch=[]
    for i in range(6):
        all_patch.append(torch.load(patch_files[i])['img'])
    for i in all_patch:
        print(i.shape)
    shape=i.shape
    sample_p=[]
    sample_n=[]
    for x in range(shape[1]):
        for y in range(shape[2]):
            #损毁建筑，取pre和post,否则取pre和median
            if mask[x,y]>0.5:
                #不使用有空值的图像
                if np.isnan(all_patch[0][:,x,y]).any() or np.isnan(all_patch[2][:,x,y]).any():
                    continue
                else:
                    sample_p.append([[x,y],
                                    np.stack([all_patch[0][:,x,y],all_patch[2][:,x,y]],axis=0),
                                    np.stack([all_patch[3][:,x,y],all_patch[5][:,x,y]],axis=0)])
            else:
                if np.isnan(all_patch[0][:,x,y]).any() or np.isnan(all_patch[1][:,x,y]).any():
                    continue
                else:
                    sample_n.append([[x,y],
                                    np.stack([all_patch[0][:,x,y],all_patch[1][:,x,y]],axis=0),
                                    np.stack([all_patch[3][:,x,y],all_patch[4][:,x,y]],axis=0),])

    return sample_p,sample_n
# %%
sample_p,sample_n=get_sample(point_file,patch_files)
# %%
torch.save(sample_p,f'../data/sample/sample_sentinel2/Livoberezhyny_P_size{size}_Destroyed.pth.tar')
torch.save(sample_n,f'../data/sample/sample_sentinel2/Livoberezhyny_N_size{size}_Destroyed.pth.tar')
# %%
print(len(sample_p),len(sample_n))
# %%
