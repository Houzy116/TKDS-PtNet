#%%
import numpy as np
import os
import sys
import torch
import gdal
from torch.utils.data import DataLoader
from torchvision import utils,transforms
from torch.utils import data
from .transform import *
import sys
sys.path.append('..')
import config
from glob import glob
import imageio
import random
#%%
from sklearn.preprocessing import StandardScaler


#%%

def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_seed(seed=1234):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_loader(train=True,val=True):
    cfg=config.cfg
    DC=config.DataConfig()
    train_dir=DC.info['train_dir']
    val_dir=DC.info['test_dir']
    print(train_dir)
    print(val_dir)
    setup_seed()
    dataset_name=DC.name
    if train:
        training_set=eval(cfg['loader']['name']+'_Dataset')(cfg,dir=train_dir,
                                                            is_train=True)
    else:
        train_set=None
    if val:
        val_set=eval(cfg['loader']['name']+'_Dataset')(cfg,dir=val_dir,
                                                         is_train=False)
        # print(val_set)
    else:
        val_set=None
    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=cfg['loader']['batch_size'],
                                 shuffle=cfg['loader']['shuffle'], num_workers=cfg['loader']['num_workers'],worker_init_fn=_init_fn)
                   for x in ['train', 'val']}
    # dataloaders = {x: DataLoader(datasets[x], batch_size=cfg['loader']['batch_size'],
    #                              shuffle=cfg['loader']['shuffle'], num_workers=cfg['loader']['num_workers'],worker_init_fn=_init_fn)
    #                for x in ['train', 'val']}
    if cfg['v_generalization'][0]:
        gs_set=[]
        for city in ['Aleppo','Hama','Homs','Raqqa','Deir-ez-Zor','Idlib']:
            eval('DC.'+cfg['dataset']['name'])(cfg['dataset']['args']['size'],city,cfg['dataset']['args']['mode'])
            ds=eval(cfg['loader']['name']+'_Dataset')(cfg,dir=DC.info['test_dir'],
                                                         is_train=False)
            # print(DC.info['test_dir'])
            gs_set.append(ds)
        g_dataloaders=[DataLoader(x, batch_size=cfg['loader']['batch_size'],
                                 shuffle=cfg['loader']['shuffle'], num_workers=cfg['loader']['num_workers'],worker_init_fn=_init_fn) for x in gs_set]
        dataloaders['g']=g_dataloaders
    return dataloaders

def get_rebuiling_loader(rebuilding_split_path,batch_size=8):
    cfg=config.cfg
    setup_seed()
    val_set=eval(cfg['loader']['name']+'_Dataset')(cfg,dir=rebuilding_split_path,
                                                        is_train=False)
    dataloaders=DataLoader(val_set, batch_size=batch_size,
                                 shuffle=cfg['loader']['shuffle'], num_workers=cfg['loader']['num_workers'],worker_init_fn=_init_fn)
    return dataloaders                            

class sentinel2_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(sentinel2_Dataset,self).__init__()
        self.dir=dir
        self.img_name_list=glob(os.path.join(self.dir,'*.pth.tar'))
        self.length=len(self.img_name_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform2'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True
    def __getitem__(self,index):
        path=self.img_name_list[index]
        data=torch.load(path)
        _,img_10m,img_20m=data
        label=os.path.basename(path)[0]
        label_v=1 if label=='P' else 0
        img_10m=img_10m.transpose(1,2,0)
        img_20m=img_20m.transpose(1,2,0)
        if self.hv_flip_together:
            img_10m,img_20m=hv_flip_together(img_10m,img_20m)
        img_10m=self.transform(img_10m.copy())
        img_20m=self.transform(img_20m.copy())

        return [img_10m,img_20m,label_v]
    def __len__(self):
	    return len(self.img_name_list)

class Usize6_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(Usize6_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True     
        self.mean=[[1117.9698455633757,
                    992.3849579428162,
                    893.4330382327423,
                    1912.3388423964425,
                    3396.7250214954615,
                    3153.6224224898942,
                    3232.2098774077836,
                    3711.400667137427],
                    [1088.7133555874584,
                    1765.4918492321938,
                    2029.711742941513,
                    2180.290725632317,
                    1553.5621520361358,
                    1023.8012535452922,
                    3365.1232380529973,
                    3663.225712073641,
                    3818.539711163245,
                    3909.9734156580334,
                    3565.4635491438776,
                    3079.9576218432794]
                    ]  
        self.std=[ [219.46961630812876,
                    241.8639145705645,
                    336.4126804693181,
                    477.90657236318106,
                    1844.0165590911697,
                    1750.344612471887,
                    1963.8997391067082,
                    1906.7644338094058],
                    [266.85794015242925,
                    348.388521694905,
                    431.3354105914535,
                    489.7425412691854,
                    353.3967170515484,
                    293.1085865347951,
                    1943.7862135959344,
                    1919.0597173713863,
                    1978.1437944548468,
                    2008.3794752017525,
                    1667.075975068135,
                    1394.785668311598]
                    ]
    def norm(self,img,re='10'):
        if re=='10':
            mean=self.mean[0]
            std=self.std[0]
        else:
            mean=self.mean[1]
            std=self.std[1]
        mean=np.expand_dims(np.array(mean), 1)
        mean=np.expand_dims(mean, 1)
        std=np.expand_dims(np.array(std), 1)
        std=np.expand_dims(std, 1)

        return (img-mean)/std
    def __getitem__(self,index):
        path,label=self.sample_list[index]
        _,_,img10,img20=torch.load(path)
        img10=self.norm(img10,'10')
        img20=self.norm(img20,'20')
        img10,img20=img10.transpose(1,2,0),img20.transpose(1,2,0)
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path)
            print('label error')
        if self.hv_flip_together:
            img10,img20=hv_flip_together(img10,img20)
        img10=self.transform(img10.copy())
        img20=self.transform(img20.copy())
        return [img10,img20,label_v]
    def __len__(self):
	    return len(self.sample_list)

class Usize120_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(Usize120_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True     
        self.mean=[[1117.9698455633757,
                    992.3849579428162,
                    893.4330382327423,
                    1912.3388423964425,
                    3396.7250214954615,
                    3153.6224224898942,
                    3232.2098774077836,
                    3711.400667137427],
                    [1088.7133555874584,
                    1765.4918492321938,
                    2029.711742941513,
                    2180.290725632317,
                    1553.5621520361358,
                    1023.8012535452922,
                    3365.1232380529973,
                    3663.225712073641,
                    3818.539711163245,
                    3909.9734156580334,
                    3565.4635491438776,
                    3079.9576218432794]
                    ]  
        self.std=[ [219.46961630812876,
                    241.8639145705645,
                    336.4126804693181,
                    477.90657236318106,
                    1844.0165590911697,
                    1750.344612471887,
                    1963.8997391067082,
                    1906.7644338094058],
                    [266.85794015242925,
                    348.388521694905,
                    431.3354105914535,
                    489.7425412691854,
                    353.3967170515484,
                    293.1085865347951,
                    1943.7862135959344,
                    1919.0597173713863,
                    1978.1437944548468,
                    2008.3794752017525,
                    1667.075975068135,
                    1394.785668311598]
                    ]
    def norm(self,img,re='10'):
        if re=='10':
            mean=self.mean[0]
            std=self.std[0]
        else:
            mean=self.mean[1]
            std=self.std[1]
        mean=np.expand_dims(np.array(mean), 1)
        mean=np.expand_dims(mean, 1)
        std=np.expand_dims(np.array(std), 1)
        std=np.expand_dims(std, 1)

        return (img-mean)/std
    def __getitem__(self,index):
        path,label=self.sample_list[index]
        _,_,img10,img20=torch.load(path)
        img10=self.norm(img10,'10')
        img20=self.norm(img20,'20')
        img_10=torch.nn.functional.interpolate(torch.tensor(img10).unsqueeze_(0), size=[120, 120], mode='area')
        img_20=torch.nn.functional.interpolate(torch.tensor(img20).unsqueeze_(0), size=[120, 120], mode='area')
        # print(img_10.shape)
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path)
            print('label error')
        if self.hv_flip_together:
            img10,img20=hv_flip_together(img10,img20)
        img_10=self.transform(np.array(img_10[0]).transpose(1,2,0).copy())
        img_20=self.transform(np.array(img_20[0]).transpose(1,2,0).copy())
        img=np.concatenate((img_10,img_20),axis=0)
        return [img,label_v]
    def __len__(self):
	    return len(self.sample_list)

class Usize6RGB_Dataset(Usize6_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(Usize6RGB_Dataset,self).__init__(cfg,dir,is_train)
    def __getitem__(self,index):
        path,label=self.sample_list[index]
        _,_,img10,img20=torch.load(path)
        img10=self.norm(img10,'10')
        img10=img10.transpose(1,2,0)
        imgpre=img10[:,:,:3]
        imgpost=img10[:,:,4:7]
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path)
            print('label error')


        if self.hv_flip_together:
            imgpre,imgpost=hv_flip_together(imgpre,imgpost)
        imgpre=self.transform(imgpre.copy())
        imgpost=self.transform(imgpost.copy())
        img=np.concatenate([imgpre,imgpost],axis=0)
        return [img,label_v]

class Usize6multitime_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(Usize6multitime_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        self.is_train=is_train
        self.cfg=cfg
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True     
        self.mean=[[1117.9698455633757,
                    992.3849579428162,
                    893.4330382327423,
                    1912.3388423964425,
                    3396.7250214954615,
                    3153.6224224898942,
                    3232.2098774077836,
                    3711.400667137427],
                    [1088.7133555874584,
                    1765.4918492321938,
                    2029.711742941513,
                    2180.290725632317,
                    1553.5621520361358,
                    1023.8012535452922,
                    3365.1232380529973,
                    3663.225712073641,
                    3818.539711163245,
                    3909.9734156580334,
                    3565.4635491438776,
                    3079.9576218432794]
                    ]  
        self.std=[ [219.46961630812876,
                    241.8639145705645,
                    336.4126804693181,
                    477.90657236318106,
                    1844.0165590911697,
                    1750.344612471887,
                    1963.8997391067082,
                    1906.7644338094058],
                    [266.85794015242925,
                    348.388521694905,
                    431.3354105914535,
                    489.7425412691854,
                    353.3967170515484,
                    293.1085865347951,
                    1943.7862135959344,
                    1919.0597173713863,
                    1978.1437944548468,
                    2008.3794752017525,
                    1667.075975068135,
                    1394.785668311598]
                    ]
    def norm(self,img,re='10'):
        if re=='10':
            mean=self.mean[0]
            std=self.std[0]
        else:
            mean=self.mean[1]
            std=self.std[1]
        mean=np.expand_dims(np.array(mean), 1)
        mean=np.expand_dims(mean, 1)
        std=np.expand_dims(np.array(std), 1)
        std=np.expand_dims(std, 1)

        return (img-mean)/std
    def enhance(self,path_post_,labels_,mode):
        l=len([i for i in labels_ if i is not None])
        if mode=='RandomDel':
            if self.is_train:
                path_post=[]
                labels=[]
                if l>3:
                    mask_num=random.randint(0,l-3)#最少留3个
                    mask=random.sample(range(0, l), mask_num)
                    # print(len(labels_))
                    # print(len(mask))
                    for i in range(len(path_post_)):
                        if i not in mask:
                            path_post.append(path_post_[i])
                            labels.append(labels_[i])

                    path_post+=[None]*len(mask)
                    labels+=[None]*len(mask)
                else:
                    path_post=path_post_
                    labels=labels_                    
                # print(len(labels))
                # path_post=path_post_
                # labels=labels_
            else:
                path_post=path_post_
                labels=labels_
        elif mode=='LabelMask':
            if self.is_train:
                if l>3:
                    mask_num=random.randint(0,l-3)#最少留3个 
                    mask=random.sample(range(0, l), mask_num)
                    # print(len(labels_))
                    # print(len(mask))
                    for i in mask:
                        labels_[i]=None
                
                # print(len(labels))
                path_post=path_post_
                labels=labels_
            else:
                path_post=path_post_
                labels=labels_
        elif mode=='Shuffle':
            if self.is_train:

                index=list(range(l))
                random.shuffle(index)
                path_post=[path_post_[i] for i in index]
                labels=[labels_[i] for i in index]
                path_post+=[None]*(20-l)
                labels+=[None]*(20-l)

            else:
                path_post=path_post_
                labels=labels_
        elif mode=='RandomDelandCopy':
            if self.is_train:
                path_post=[]
                labels=[]
                if l>3:
                    del_num=random.randint(0,l-3)#最少留3个
                    # copy_num=random.randint(0,del_num)
                    copy_num=del_num
                    index=list(range(l))
                    copy_index=random.sample(range(0, l), copy_num)
                    index+=copy_index
                    index.sort()
                    del_index=random.sample(range(0, len(index)), del_num)
                    index_=[index[i] for i in range(len(index)) if i not in del_index]

                    for i in index_:
                        path_post.append(path_post_[i])
                        labels.append(labels_[i])

                    path_post+=[None]*(20-len(path_post))
                    labels+=[None]*(20-len(labels))
                else:
                    path_post=path_post_
                    labels=labels_                    
                # print(len(labels))
                # path_post=path_post_
                # labels=labels_
        elif mode=='RandomDelandCopyv2':
            if self.is_train:
                path_post=[]
                labels=[]
                # if l>3:
                    
                # copy_num=random.randint(0,del_num)
                if l<20-l:
                    copy_num=random.randint(0,l)
                else:
                    copy_num=random.randint(0,20-l)
                if l+copy_num-5>0:
                    del_num=random.randint(0,l+copy_num-5)#最少留5个
                else:
                    del_num=0

                index=list(range(l))
                copy_index=random.sample(range(0, l), copy_num)
                index+=copy_index
                index.sort()
                del_index=random.sample(range(0, len(index)), del_num)
                index_=[index[i] for i in range(len(index)) if i not in del_index]

                for i in index_:
                    path_post.append(path_post_[i])
                    labels.append(labels_[i])

                path_post+=[None]*(20-len(path_post))
                labels+=[None]*(20-len(labels))
                # else:
                #     path_post=path_post_
                #     labels=labels_                    
                # print(len(labels))
                # path_post=path_post_
                # labels=labels_
            else:
                path_post=path_post_
                labels=labels_   
        elif mode is None:
            if self.is_train:
                path_post=path_post_
                labels=labels_
            else:
                path_post=path_post_
                labels=labels_
        return path_post,labels

    def load_img(self,path):
        _,_,img10,img20=torch.load(path)
        img10=self.norm(img10,'10')
        img20=self.norm(img20,'20')
        img10,img20=img10.transpose(1,2,0),img20.transpose(1,2,0)
        if self.hv_flip_together:
            img10,img20=hv_flip_together(img10,img20)
        img10=self.transform(img10.copy())
        img20=self.transform(img20.copy())
        return [img10,img20]

    def __getitem__(self,index):
        path_post_=self.sample_list[index][0]
        labels_=self.sample_list[index][1]
        mode=self.cfg['loader']['mode']
        path_post,labels=self.enhance(path_post_,labels_,mode)
        
        imgs=[self.load_img(path) if path is not None else [np.zeros((8,6,6)),np.zeros((12,3,3))] for path in path_post]
        data10=np.concatenate([i[0] for i in imgs],axis=0)
        data20=np.concatenate([i[1] for i in imgs],axis=0)

        labels_v=[]
        for label in labels:
            if label=='p':
                labels_v.append(1)
            elif label=='n':
                labels_v.append(0)
            elif label is None:
                labels_v.append(-1)
            else:
                print(label)
                print(path_post[0])
                print('label error')
        if np.array(labels_v).sum()==-20:
            print(labels_)
            print(labels)
            # print(mask_num,mask)
        labels_v=torch.tensor(labels_v)

        return [data10,data20,labels_v]
    def __len__(self):
	    return len(self.sample_list)

class sentinel2class_Dataset(sentinel2_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(sentinel2class_Dataset,self).__init__(cfg,dir,is_train)
    def __getitem__(self,index):
        path=self.img_name_list[index]
        data=torch.load(path)
        _,img_10m,img_20m=data
        label=os.path.basename(path).split('_')[0]
        if label=='P1':
            label_v=1
        elif label=='P2':
            label_v=2
        elif label=='N':
            label_v=0
        else:
            print(label)
            print(path)
            print('label error')
        img_10m=img_10m.transpose(1,2,0)
        img_20m=img_20m.transpose(1,2,0)
        if self.hv_flip_together:
            img_10m,img_20m=hv_flip_together(img_10m,img_20m)
        img_10m=self.transform(img_10m.copy())
        img_20m=self.transform(img_20m.copy())
        return [img_10m,img_20m,label_v]
    def __len__(self):
	    return len(self.img_name_list)

class google_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(google_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True

    def __getitem__(self,index):
        path_pre=self.sample_list[index][0]
        path_post=path_pre.replace('pre','post')
        img_pre=imageio.imread(path_pre)
        img_post=imageio.imread(path_post)
        

        label=self.sample_list[index][1]
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path_pre)
            print('label error')
        if self.hv_flip_together:
            img_pre,img_post=hv_flip_together(img_pre,img_post)
        img_pre=self.transform(img_pre.copy())
        img_post=self.transform(img_post.copy())
        data=np.concatenate((img_pre,img_post),axis=0)
        return [data,label_v,path_pre]
    def __len__(self):
	    return len(self.sample_list)
    
class googlesize6_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(googlesize6_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True

    def __getitem__(self,index):
        path_pre=self.sample_list[index][0]
        path_post=path_pre.replace('pre','post')
        img_pre=torch.load(path_pre).transpose(1,2,0)
        img_post=torch.load(path_post).transpose(1,2,0)
        

        label=self.sample_list[index][1]
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path_pre)
            print('label error')
        if self.hv_flip_together:
            img_pre,img_post=hv_flip_together(img_pre,img_post)
        img_pre=self.transform(img_pre.copy())
        img_post=self.transform(img_post.copy())
        data=np.concatenate((img_pre,img_post),axis=0)
        return [data,label_v]
    def __len__(self):
	    return len(self.sample_list)

class googlesize6upresample_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(googlesize6upresample_Dataset,self).__init__()
        print(dir)
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True

    def __getitem__(self,index):
        path_pre=self.sample_list[index][0]
        path_post=path_pre.replace('pre','post')
        img_pre=torch.load(path_pre).astype(np.float64)
        img_post=torch.load(path_post).astype(np.float64)
        # print(img_pre)
        # print(img_post)
        img_pre=torch.nn.functional.interpolate(torch.tensor(img_pre).unsqueeze_(0), size=[120, 120], mode='area')
        img_post=torch.nn.functional.interpolate(torch.tensor(img_post).unsqueeze_(0), size=[120, 120], mode='area')
        img_pre=np.array(img_pre[0]).transpose(1,2,0)
        img_post=np.array(img_post[0]).transpose(1,2,0)
        # print('post',img_post.shape,'pre',img_pre.shape)
        label=self.sample_list[index][1]
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path_pre)
            print('label error')
        if self.hv_flip_together:
            img_pre,img_post=hv_flip_together(img_pre,img_post)
        img_pre=self.transform(img_pre.copy())
        img_post=self.transform(img_post.copy())
        data=np.concatenate((img_pre,img_post),axis=0)
        return [data,label_v]
    def __len__(self):
	    return len(self.sample_list)

class googlesize6multitime_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(googlesize6multitime_Dataset,self).__init__()
        self.cfg=config.cfg
        print(dir)
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        self.is_train=is_train

        self.std_scaler = StandardScaler()
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True

    def enhance(self,path_post_,labels_,mode,length):
        l=len([i for i in labels_ if i is not None])
        if mode=='RandomDel':
            if self.is_train:
                path_post=[]
                labels=[]
                if l>3:
                    mask_num=random.randint(0,l-3)#最少留3个
                    mask=random.sample(range(0, l), mask_num)
                    # print(len(labels_))
                    # print(len(mask))
                    for i in range(len(path_post_)):
                        if i not in mask:
                            path_post.append(path_post_[i])
                            labels.append(labels_[i])

                    path_post+=[None]*len(mask)
                    labels+=[None]*len(mask)
                else:
                    path_post=path_post_
                    labels=labels_                    
                # print(len(labels))
                # path_post=path_post_
                # labels=labels_
            else:
                path_post=path_post_
                labels=labels_
        elif mode=='LabelMask':
            if self.is_train:
                if l>3:
                    mask_num=random.randint(0,l-3)#最少留3个 
                    mask=random.sample(range(0, l), mask_num)
                    # print(len(labels_))
                    # print(len(mask))
                    for i in mask:
                        labels_[i]=None
                
                # print(len(labels))
                path_post=path_post_
                labels=labels_
            else:
                path_post=path_post_
                labels=labels_
        elif mode=='Shuffle':
            if self.is_train:

                index=list(range(l))
                random.shuffle(index)
                path_post=[path_post_[i] for i in index]
                labels=[labels_[i] for i in index]
                path_post+=[None]*(length-l)
                labels+=[None]*(length-l)

            else:
                path_post=path_post_
                labels=labels_
        elif mode=='RandomDelandCopy':
            if self.is_train:
                path_post=[]
                labels=[]
                if l>3:
                    del_num=random.randint(0,l-3)#最少留3个
                    # copy_num=random.randint(0,del_num)
                    copy_num=del_num
                    index=list(range(l))
                    copy_index=random.sample(range(0, l), copy_num)
                    index+=copy_index
                    index.sort()
                    del_index=random.sample(range(0, len(index)), del_num)
                    index_=[index[i] for i in range(len(index)) if i not in del_index]

                    for i in index_:
                        path_post.append(path_post_[i])
                        labels.append(labels_[i])

                    path_post+=[None]*(length-len(path_post))
                    labels+=[None]*(length-len(labels))
                else:
                    path_post=path_post_
                    labels=labels_                    
                # print(len(labels))
                # path_post=path_post_
                # labels=labels_
        elif mode=='RandomDelandCopyv2':
            if self.is_train:
                path_post=[]
                labels=[]
                # if l>3:
                    
                # copy_num=random.randint(0,del_num)
                if l<length-l:
                    copy_num=random.randint(0,l)
                else:
                    copy_num=random.randint(0,length-l)
                if l+copy_num-5>0:
                    del_num=random.randint(0,l+copy_num-5)#最少留3个
                else:
                    del_num=0

                index=list(range(l))
                copy_index=random.sample(range(0, l), copy_num)
                index+=copy_index
                index.sort()
                del_index=random.sample(range(0, len(index)), del_num)
                index_=[index[i] for i in range(len(index)) if i not in del_index]

                for i in index_:
                    path_post.append(path_post_[i])
                    labels.append(labels_[i])

                path_post+=[None]*(length-len(path_post))
                labels+=[None]*(length-len(labels))
                # else:
                #     path_post=path_post_
                #     labels=labels_                    
                # print(len(labels))
                # path_post=path_post_
                # labels=labels_
            else:
                path_post=path_post_
                labels=labels_   
        elif mode is None:
            if self.is_train:
                path_post=path_post_
                labels=labels_
            else:
                path_post=path_post_
                labels=labels_
        return path_post,labels
    def cat(self,img_pre,img_post):

        if self.hv_flip_together:
            img_pre,img_post=hv_flip_together(img_pre.transpose(1,2,0),img_post.transpose(1,2,0))
        else:
            img_pre,img_post=img_pre.transpose(1,2,0),img_post.transpose(1,2,0)
        # print(41,img_pre.shape)
        # print(42,img_post.shape)
        img_pre=self.transform(img_pre.copy())
        img_post=self.transform(img_post.copy())
        # print(51,img_pre.shape)
        # print(52,img_post.shape)
        data=np.concatenate((img_pre,img_post),axis=0)
        
        return data
    def __getitem__(self,index):
        path_pre=self.sample_list[index][0][0]
        path_post_=self.sample_list[index][0][1:]
        labels_=self.sample_list[index][1][1:]
        length=16
        mode=self.cfg['loader']['mode']
        path_post,labels=self.enhance(path_post_,labels_,mode,length)

        img_pre=torch.load(path_pre)
        # print(1,img_pre.shape)
        imgs=[self.cat(img_pre,torch.load(post)) if post is not None else np.zeros((6,6,6)) for post in path_post]
        if self.cfg['v_generalization'][0]:
            if self.cfg['v_generalization'][1]=='allnormalized':
                for i in range(len(imgs)):
                    img=imgs[i].reshape([6,-1])
                    self.std_scaler.fit(img)
                    imgs[i]=self.std_scaler.transform(img).reshape([6,6,-1])
            elif self.cfg['v_generalization'][1]=='channelsnormalized':
                for i in range(len(imgs)):
                    img=imgs[i]
                    for c in range(6):
                        self.std_scaler.fit(img[c,:,:])
                        img[c,:,:]=self.std_scaler.transform(img[c,:,:])
                    imgs[i]=img
            elif self.cfg['v_generalization'][1]=='channels3normalized':
                for i in range(len(imgs)):
                    pre=imgs[i][:3,:,:].reshape([3,-1])
                    post=imgs[i][3:,:,:].reshape([3,-1])
                    self.std_scaler.fit(pre)
                    imgs[i][:3,:,:]=self.std_scaler.transform(pre).reshape([3,6,-1])
                    self.std_scaler.fit(post)
                    imgs[i][3:,:,:]=self.std_scaler.transform(post).reshape([3,6,-1])
        # print(2,len(imgs))
        # print(3,imgs[0].shape)
        data=np.concatenate(imgs,axis=0)


        labels_v=[]
        for label in labels:
            if label=='p':
                labels_v.append(1)
            elif label=='n':
                labels_v.append(0)
            elif label is None:
                labels_v.append(-1)
            else:
                print(label)
                print(path_pre)
                print('label error')
        if np.array(labels_v).sum()==-16:
            print(labels_)
            print(labels)
            # print(mask_num,mask)
        labels_v=torch.tensor(labels_v)
        return [data,labels_v]
    def __len__(self):
	    return len(self.sample_list)

class googlesize120multitime_Dataset(googlesize6multitime_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(googlesize120multitime_Dataset,self).__init__(cfg,dir,is_train)
    def __getitem__(self,index):
        path_pre=self.sample_list[index][0][0]
        path_post_=self.sample_list[index][0][1:]
        labels_=self.sample_list[index][1][1:]
        length=16  
        mode=self.cfg['loader']['mode']
        path_post,labels=self.enhance(path_post_,labels_,mode,length)

        img_pre=imageio.imread(path_pre).transpose(2,0,1)
        # print(1,img_pre.shape)
        imgs=[self.cat(img_pre,imageio.imread(post).transpose(2,0,1)) if post is not None else np.zeros((6,120,120)) for post in path_post]
        # for i in imgs:
        #     print(i.shape)
        # print(2,len(imgs))
        # print(3,imgs[0].shape)
        data=np.concatenate(imgs,axis=0)

        labels_v=[]
        for label in labels:
            if label=='p':
                labels_v.append(1)
            elif label=='n':
                labels_v.append(0)
            elif label is None:
                labels_v.append(-1)
            else:
                print(label)
                print(path_pre)
                print('label error')
        if np.array(labels_v).sum()==-16:
            print(labels_)
            print(labels)
            # print(mask_num,mask)
        labels_v=torch.tensor(labels_v)
        return [data,labels_v]
    def __len__(self):
	    return len(self.sample_list)

class googlesize6upresamplemultitime_Dataset(googlesize6multitime_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(googlesize6upresamplemultitime_Dataset,self).__init__(cfg,dir,is_train)
    def upresample(self,img_size6):
        img_size6=img_size6.astype(np.float64)
        img_size120=torch.nn.functional.interpolate(torch.tensor(img_size6).unsqueeze_(0),size=[120,120],mode='area')
        return np.array(img_size120[0])
    def __getitem__(self,index):
        path_pre=self.sample_list[index][0][0]
        path_post_=self.sample_list[index][0][1:]
        labels_=self.sample_list[index][1][1:]
        length=16
        mode=self.cfg['loader']['mode']
        path_post,labels=self.enhance(path_post_,labels_,mode,length)

        img_pre=self.upresample(torch.load(path_pre))

        # print(1,img_pre.shape)
        imgs=[self.cat(img_pre,self.upresample(torch.load(post))) if post is not None else np.zeros((6,120,120)) for post in path_post]
        # print(2,len(imgs))
        # print(3,imgs[0].shape)
        data=np.concatenate(imgs,axis=0)
        # if data.shape[0]>96:
        #     print(data.shape)
        #     print(path_post)
        #     print(len(imgs))
        #     print([i.shape for i in imgs])
        #     raise



        labels_v=[]
        for label in labels:
            if label=='p':
                labels_v.append(1)
            elif label=='n':
                labels_v.append(0)
            elif label is None:
                labels_v.append(-1)
            else:
                print(label)
                print(path_pre)
                print('label error')
        if np.array(labels_v).sum()==-16:
            print(labels_)
            print(labels)
            # print(mask_num,mask)
        labels_v=torch.tensor(labels_v)
        return [data,labels_v]
    def __len__(self):
	    return len(self.sample_list)

class USAsize120_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(USAsize120_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True

    def __getitem__(self,index):
        path_pre=self.sample_list[index][0]

        img_pre=np.load(path_pre).transpose(1,2,0)*100

        

        label=self.sample_list[index][1]
        if self.hv_flip_together:
            img_pre,_=hv_flip_together(img_pre,np.zeros((120,120,7)))
        img_pre=self.transform(img_pre.copy())
        return [img_pre,label]
    def __len__(self):
	    return len(self.sample_list)
        
class USAsize120multitime_Dataset(googlesize6multitime_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(USAsize120multitime_Dataset,self).__init__(cfg,dir,is_train)
    def __getitem__(self,index):
        img_pathes=self.sample_list[index][0]
        labels=self.sample_list[index][1]
        length=12
        # mode=self.cfg['loader']['mode']
        # path_post,labels=self.enhance(img_pathes,labels_,mode,length)
        # print(1,img_pre.shape)
        imgs=[np.load(post) if post is not None else np.zeros((7,120,120)) for post in img_pathes]
        # for i in imgs:
        #     print(i.shape)
        # print(2,len(imgs))
        # print(3,imgs[0].shape)
        data=np.concatenate(imgs,axis=0)

        labels_v=[]
        for label in labels:
            if label is None:
                labels_v.append(-1)
            else:
                labels_v.append(label)

            # print(mask_num,mask)
        labels_v=torch.tensor(labels_v)
        return [data,labels_v]
    def __len__(self):
	    return len(self.sample_list)

class USAsize120multitimev4_Dataset(googlesize6multitime_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(USAsize120multitimev4_Dataset,self).__init__(cfg,dir,is_train)
    def __getitem__(self,index):
        img_pathes=self.sample_list[index][0]
        labels=self.sample_list[index][1]
        length=12
        # mode=self.cfg['loader']['mode']
        # path_post,labels=self.enhance(img_pathes,labels_,mode,length)
        # print(1,img_pre.shape)
        imgs=[np.load(post) if post is not None else np.zeros((7,120,120)) for post in img_pathes]
        # for i in imgs:
        #     print(i.shape)
        # print(2,len(imgs))
        # print(3,imgs[0].shape)
        data=np.concatenate(imgs,axis=0)
        labels_v=[np.array([i[1],i[3]]) if i is not None else np.array([-1]*2) for i in labels]
        labels_v=np.stack(labels_v,axis=0)

            # print(mask_num,mask)
        labels_v=torch.tensor(labels_v)
        return [data,labels_v]
    def __len__(self):
	    return len(self.sample_list)

class CENsize120multitime_Dataset(googlesize6multitime_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(CENsize120multitime_Dataset,self).__init__(cfg,dir,is_train)
    def __getitem__(self,index):
        img_pathes=self.sample_list[index][0]
        labels=self.sample_list[index][1]
        length=5
        # mode=self.cfg['loader']['mode']
        # path_post,labels=self.enhance(img_pathes,labels_,mode,length)
        # print(1,img_pre.shape)
        imgs=[np.load(path,allow_pickle=True) if path is not None else [np.zeros((4,120,120)),np.zeros((6,60,60))] for path in img_pathes]
        data10=np.concatenate([i[0] for i in imgs],axis=0)
        data20=np.concatenate([i[1] for i in imgs],axis=0)
        labels_v=[np.array(i) if i is not None else np.array([-1]*4) for i in labels]
        labels_v=np.stack(labels_v,axis=0)
        labels_v=labels_v[:,:3]
        # imgs=[np.load(post) if post is not None else np.zeros((7,120,120)) for post in img_pathes]
        # for i in imgs:
        #     print(i.shape)
        # print(2,len(imgs))
        # print(3,imgs[0].shape)
        # data=np.concatenate(imgs,axis=0)


        return [data10,data20,labels_v]
    def __len__(self):
	    return len(self.sample_list)

class CENsize120_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(CENsize120_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True

    def __getitem__(self,index):
        path=self.sample_list[index][0]
        label=self.sample_list[index][1][:3]
        data10,data20=np.load(path,allow_pickle=True)
        label=np.array(label)
        return [data10,data20,label]
    def __len__(self):
	    return len(self.sample_list)

class googleadd3pre_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(googleadd3pre_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        self.cfg=config.cfg
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True
    def cat(self,imgs):
            # 50%的概率应用垂直，水平翻转。
        data_list=[]
        if self.hv_flip_together:
            if random.random() > 0.5:
                for i in range(len(imgs)):
                    imgs[i]=np.flip(imgs[i],axis=0)
            if random.random() > 0.5:
                for i in range(len(imgs)):
                    imgs[i]=np.flip(imgs[i],axis=1)
        for i in range(len(imgs)):
            data_list.append(self.transform(imgs[i].copy()))
        data=np.concatenate(data_list,axis=0)
        # image = tf.to_tensor(image)
        # image2 = tf.to_tensor(image2)
        return data
    def __getitem__(self,index):
        path_pre=self.sample_list[index][0]
        path_post=path_pre.replace('pre','post')

        basename=os.path.basename(path_post)
        city=basename.split('_')[0]
        coord=basename.split('_')[-1].split('.')[0]
        add3pre=glob(f'/ssd/hk/Syria_samples/samples_addpre/samples/{city}_*{coord}.png')
        if len(add3pre)!=3:
            raise('add pre img num !=3')
        add3pre.sort()
        add3pre.append(path_pre)
        add3pre.append(path_post)
        imgs=[imageio.imread(i) for i in add3pre]
        mode=self.cfg['loader']['mode']
        if mode=='mt':
            data=[]
            for i in range(len(imgs)-1):
                data+=[imgs[i]]
                data+=[imgs[-1]]
            data=torch.cat(data,axis=0)
        elif mode is None:
            data=torch.cat(imgs,axis=0)
        else:
            raise(f'no {mode} mode')

        

        label=self.sample_list[index][1]
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path_pre)
            print('label error')

        return [data,label_v]
    def __len__(self):
	    return len(self.sample_list)
    
class googleadd3presize6_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(googleadd3presize6_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        self.cfg=config.cfg
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True


    def my_trf(self,path,h,v):
        img=torch.load(path).transpose(1,2,0)
        if h:
            img=np.flip(img,axis=0)
        if v:
            img=np.flip(img,axis=1)
        img=self.transform(img.copy())
        return img
    def __getitem__(self,index):
        path_pre=self.sample_list[index][0]
        path_post=path_pre.replace('pre','post')

        basename=os.path.basename(path_post)
        city=basename.split('_')[0]
        coord=basename.split('_')[-1].split('.')[0]
        add3pre=glob(f'/ssd/hk/Syria_samples/samples_addpre_size6_AREA/samples/{city}_*{coord}.pth')
        if len(add3pre)!=3:
            raise('add pre img num !=3')
        add3pre.sort()
        add3pre.append(path_pre)
        add3pre.append(path_post)
        
        h,v=False,False
        if self.hv_flip_together:
            if random.random() > 0.5:
                h=True
            if random.random() > 0.5:
                v=True    
        imgs=[self.my_trf(i,h,v) for i in add3pre]
        mode=self.cfg['loader']['mode']
        # print(mode)
        if mode=='mt':
            data=[]
            for i in range(len(imgs)-1):
                data+=[imgs[i]]
                data+=[imgs[-1]]
            data=torch.cat(data,axis=0)
        elif mode is None:
            data=torch.cat(imgs,axis=0)
        else:
            raise(f'no {mode} mode')
        

        label=self.sample_list[index][1]
        if label=='p':
            label_v=1
        elif label=='n':
            label_v=0
        else:
            print(label)
            print(path_pre)
            print('label error')

        return [data,label_v]
    def __len__(self):
	    return len(self.sample_list)

class SEGsize120_Dataset(data.Dataset):
    def __init__(self,cfg,dir,is_train):
        super(SEGsize120_Dataset,self).__init__()
        self.sample_list=torch.load(dir)
        use_samples_percent=cfg['loader']['use_samples_percent']
        use_samples=int(len(self.sample_list)*use_samples_percent)
        self.sample_list=self.sample_list[:use_samples]
        self.length=len(self.sample_list)
        self.hv_flip_together=False
        if is_train:
            self.transform=get_transform(cfg['transform_train'])
            if 'hv_flip_together' in cfg['transform_train'].keys() and cfg['transform_train']['hv_flip_together']:
                self.hv_flip_together=True
        else:
            self.transform=get_transform(cfg['transform_val'])
            if 'hv_flip_together' in cfg['transform_val'].keys() and cfg['transform_val']['hv_flip_together']:
                self.hv_flip_together=True

    def __getitem__(self,index):
        path_img=self.sample_list[index]
        img=gdal.Open(path_img).ReadAsArray().transpose(1,2,0)#7,120,120
        path_label=path_img.replace('/img/','/label/').replace('.tif','_label.tif')
        label=gdal.Open(path_label).ReadAsArray()

        if self.hv_flip_together:
            img,label=hv_flip_together(img,label)
        img=self.transform(img.copy())
        label=self.transform(label.copy())
        return [img,label]
    def __len__(self):
	    return len(self.sample_list)

class SEGsize120multitime_Dataset(googlesize6multitime_Dataset):
    def __init__(self,cfg,dir,is_train):
        super(SEGsize120multitime_Dataset,self).__init__(cfg,dir,is_train)
    def __getitem__(self,index):
        pathes_img=self.sample_list[index]
        pathes_label=[path_img.replace('/img/','/label/').replace('.tif','_label.tif') if path_img is not None else None for path_img in pathes_img]
        # length=9

        imgs=[gdal.Open(path_img).ReadAsArray() if path_img is not None else np.zeros((7,120,120)) for path_img in pathes_img]
        labels=[gdal.Open(path_label).ReadAsArray()[np.newaxis,:] if path_label is not None else np.zeros((1,120,120)) for path_label in pathes_label]

        img_cube=np.concatenate(imgs,axis=0)
        label_cube=np.concatenate(labels,axis=0)

        return [img_cube,label_cube]
    def __len__(self):
	    return len(self.sample_list)