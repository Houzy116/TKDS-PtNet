#%%
import sys
# from Ukraine.model.load_model import get_model
import loader
# from python_script.Ukraine.loader.loader import get_rebuiling_loader
import run
import warnings


import numpy as np
warnings.filterwarnings("ignore")
loaders=loader.get_loader()
trainer=run.get_trainer()
trainer2=trainer(loaders)
# n=trainer2.model()

# %%
# rebuilding_split_path='/ssd/hk/Syria_samples/split_havedamaged_size6_all_v3_rebuilding/val.pth'
# rebuilding_loader=loader.get_rebuiling_loader(rebuilding_split_path,batch_size=4)
# trainer.train_models(rebuilding_loader)
trainer2.train_models()
trainer2.test_model()
# trainer2.save_gt_and_pred()
trainer2.PR_figure(PR_neme=f'bestmodel_PR_curve_val.jpg')
trainer2.AUC_figure(AUC_neme=f'bestmodel_AUC_curve_val.jpg')
# trainer2.PR_figure(PR_neme=f'bestmodel_PR_curve.jpg')
# trainer2.AUC_figure(AUC_neme=f'bestmodel_AUC_curve.jpg')
#%%
len(loaders['train'])
# ['M','Rubizhne','Sievierodoetsk','volnovskha']
# p=f'/ssd/hk/Ukraine_samples/split_coordv2_v2/cities/M/val2.pth'
# l=loader.get_rebuiling_loader(p,batch_size=64)
# trainer2.logger.write(f'++++++++++++++++++ val  M ++++++++++++++++++\n')
# loaders['val']=l
# trainer2=trainer(loaders)
# trainer2.test_model()
# p=f'/ssd/hk/Ukraine_samples/split_coordv2_v2/cities/Rubizhne/val.pth'
# l=loader.get_rebuiling_loader(p,batch_size=64)
# trainer2.logger.write(f'++++++++++++++++++ val  Rubizhne ++++++++++++++++++\n')
# loaders['val']=l
# trainer2=trainer(loaders)
# trainer2.test_model()
# p=f'/ssd/hk/Ukraine_samples/split_coordv2_v2/cities/Sievierodoetsk/val.pth'
# l=loader.get_rebuiling_loader(p,batch_size=64)
# trainer2.logger.write(f'++++++++++++++++++ val  Sievierodoetsk ++++++++++++++++++\n')
# loaders['val']=l
# trainer2=trainer(loaders)
# trainer2.test_model()
# p=f'/ssd/hk/Ukraine_samples/split_coordv2_v2/cities/volnovskha/val.pth'
# l=loader.get_rebuiling_loader(p,batch_size=64)
# trainer2.logger.write(f'++++++++++++++++++ val  volnovskha ++++++++++++++++++\n')
# loaders['val']=l
# trainer2=trainer(loaders)
# trainer2.test_model()

# #%%
# p=f'/ssd/hk/Ukraine_samples/split_coordv2_v3/cities/Rubizhne/train.pth'
# l=loader.get_rebuiling_loader(p,batch_size=64)
# trainer2.logger.write(f'++++++++++++++++++ val  Rubizhne ++++++++++++++++++\n')
# loaders['val']=l
# trainer2=trainer(loaders)
# trainer2.test_model()
# #%%
# for k in ['Aleppo','Hama','Homs','Raqqa','Deir-ez-Zor','Idlib']:
# # k='Aleppo'
#     p=f'/ssd/hk/Syria_samples/split_havedamaged_size120_all_v3/cities/{k}/train.pth'
#     l=loader.get_rebuiling_loader(p,batch_size=8)
#     trainer2.logger.write(f'++++++++++++++++++ val {k} ++++++++++++++++++\n')
#     loaders['val']=l
#     trainer2=trainer(loaders)
#     trainer2.test_model()
#     trainer2.PR_figure(PR_neme=f'bestmodel_PR_curve_val_{k}.jpg')
#     trainer2.AUC_figure(AUC_neme=f'bestmodel_AUC_curve_val_{k}.jpg')
#%%














# rebuilding_split_path='/ssd/hk/Syria_samples/split_havedamaged_size6_all_v3_rebuilding/val.pth'
# rebuilding_loader=loader.get_rebuiling_loader(rebuilding_split_path,batch_size=256)
# print(len(rebuilding_loader))
# trainer.test_rebuilding(rebuilding_loader)
# # trainer.model
# # pred_=trainer.pred_
# # pred=pred_.view(-1,2)
# #%%
# loaders['val']
for i,v in enumerate(loaders['val']):
    # print(v)
    break
print(v[0].shape,v[1].shape)
print(v[1])
# import torch
# print(torch.argmax(trainer.pred,dim=1))
# print(trainer.gt)
# %%
# trainer.load_checkpoint()
# import torch
# model=trainer2.model
# pred=model(v[0].type(torch.FloatTensor).cuda())
# pred
# for i,v in enumerate(loaders['train']):
#     if i>13:
#         break
# print(v[0].shape)
# print(v[1].shape)
# model([v[0].type(torch.FloatTensor).cuda(),v[1].type(torch.FloatTensor).cuda()]).shape
# v[1].shape
# pr=model(v[0].type(torch.FloatTensor).cuda())
# out=torch.argmax(pr.view(-1,2),dim=1)
# print(np.array([out[i].cpu() for i in range(v[1].flatten().shape[0]) if v[1].flatten()[i]!=-1]))
# print(np.array([i for i in v[1].flatten() if i!=-1]))
# # %%
# # v[1]
# # # %%
# # import matplotlib.pyplot as plt
# # i=6
# # print(v[1][i])
# # plt.imshow(np.array(v[0][i,:3,:,:]).transpose((1,2,0)))
# # plt.show()
# # plt.imshow(np.array(v[0][i,3:,:,:]).transpose((1,2,0)))
# # plt.show()
# # # %%
# # v[1]
# # %%
# trainer.model
# # %%
# trainer.batch
# # %%
# v
# # %%
# trainer.model(v[0].to(trainer.device)).argmax(axis=1)
# # %%
# v[0].shape 
# # %%


# # %%
# import config
# cfg=config.cfg
# cfg
# # %%
# resnet9v2=trainer.model
# # %%
# import torch
# from collections import OrderedDict
# ck=torch.load('/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/HamaandRaqqa_size6v2_models/SupCon_HamaandRaqqa_size6v2_resnet9_lr_0.05_decay_0.0001_bsz_1024_temp_0.07_trial_0_cosine_warm/last.pth')

# w2={}
# type(ck['model'])
# if len(cfg['train']['gpu_ids'])==1:
#     for k in ck['model'].keys():
#         if 'encoder.' in k:
#             k2=k.replace('encoder.module.','')
#             w2[k2]=ck['model'][k]
# else:
#     for k in ck['model'].keys():
#         if 'encoder.' in k:
#             k2=k.replace('encoder.','')
#             w2[k2]=ck['model'][k]
# w2=OrderedDict(w2)
# # %%
# import torch.nn as nn
# resnet9v2.cuda()
# # resnet9v2 = nn.DataParallel(resnet9v2)
# # resnet9v2.load_state_dict(w2,False)
# # %%
# # print(v[0].shape,v[1].shape)
# # # %%
# # v[1].shape
# # # %%
# # gt_=v[1].flatten()
# # # %%
# # import torch
# # gt=torch.tensor([i for i in list(gt_) if i !=-1])
# # # %%
# # gt
# # # %%
# # trainer.pred.shape
# # # %%
# # trainer.gt.shape
# # # %%
# # loaders['train'].dirs
# # # %%

# # # %%
# # for i,v in enumerate(loaders['val']):
# #     if v[0].shape[1]!=15:
# #         raise
# #     break


# # # %%
# # v[0].shape
# # # %%
# v[1][35]
# # %%
# k=[]
# for i in range(256):
#     if v[1][i]==1:
#         k.append(i)
# k
# # %%
# v[1][51]
# # %%
# v[0][51].shape
# # %%
# import matplotlib.pyplot as plt
# # %%
# for z in k:
#     plt.imshow(np.array(v[0][z][:3,:,:]).transpose((1,2,0)))
#     plt.show()
#     plt.imshow(np.array(v[0][z][3:,:,:]).transpose((1,2,0)))
#     plt.show()
# # %%
# z=k[-2]
# plt.figure(figsize=(10,10))
# plt.imshow(np.array(v[0][z][:3,:,:]).transpose((1,2,0)))
# plt.show()
# plt.figure(figsize=(10,10))
# plt.imshow(np.array(v[0][z][3:,:,:]).transpose((1,2,0)))
# plt.show()
# # %%
# import torch
# img6=torch.nn.functional.interpolate(v[0][z].unsqueeze_(0), size=[6, 6], mode='area')
# # %%
# plt.figure(figsize=(10,10))
# plt.imshow(np.array(img6[0,:3,:,:]).transpose((1,2,0)))
# plt.show()
# plt.figure(figsize=(10,10))
# plt.imshow(np.array(img6[0,3:,:,:]).transpose((1,2,0)))
# plt.show()

# %%
# a=np.array([1,2,3,4])
# a=np.expand_dims(a, 1)
# a=np.expand_dims(a, 1)
# b=np.array([1,10,100,1000])
# b=np.expand_dims(b, 1)
# b=np.expand_dims(b, 1)
# (np.zeros((4,6,6))-a)/b
# %%
gt_=v[1].flatten().unsqueeze(1)
gt_.shape
# pred.contiguous().view(-1,1).shape
# v[0].shape
# %%
pred.contiguous().view(-1,1)[gt_!=-1].view(-1,1).flatten().shape
# %%
torch.tensor([i for i in list(gt_) if i !=-1]).long().shape
# %%
loss=torch.nn.MSELoss(reduction='mean')
loss(pred.contiguous().view(-1,1)[gt_!=-1].view(-1,1).type(torch.FloatTensor).flatten().cuda(),torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).cuda())
# %%
pred.contiguous().view(-1,1)[gt_!=-1].view(-1,1)
# %%
trainer2.gt.shape
# %%
trainer2.pred.shape#160,1
# %%
pred.contiguous().view(-1,1)[gt_!=-1].view(-1,1).type(torch.FloatTensor).cuda().shape
# %%
v[0].shape

# %%
v[1].shape
# %%
for i,v in enumerate(loaders['val']):
    # print(v)
    break
# v[2].shape
import torch
img=[v[0].type(torch.FloatTensor).cuda(),
    v[1].type(torch.FloatTensor).cuda()]
# %%

gt_all=v[2]
gt_0=gt_all[:,:,0].flatten()#向下取整
gt_1=gt_all[:,:,1].flatten()
gt_2=gt_all[:,:,2].flatten()
gt0=torch.tensor([i for i in list(gt_0) if i !=-1]).type(torch.FloatTensor).cuda()
gt1=torch.tensor([i for i in list(gt_1) if i !=-1]).type(torch.FloatTensor).cuda()
gt2=torch.tensor([i for i in list(gt_2) if i !=-1]).type(torch.FloatTensor).cuda()
pred_=trainer2.model(img)
pred_0=pred_[:,:,0].flatten()[gt_0!=-1].type(torch.FloatTensor).cuda()#向下取整
pred_1=pred_[:,:,1].flatten()[gt_0!=-1].type(torch.FloatTensor).cuda()
pred_2=pred_[:,:,2].flatten()[gt_0!=-1].type(torch.FloatTensor).cuda()
batch_len=pred_0.shape[0]
l=loss(pred_0.unsqueeze(1),gt0.unsqueeze(1))+loss(pred_1.unsqueeze(1),gt1.unsqueeze(1))+loss(pred_2.unsqueeze(1),gt2.unsqueeze(1))
# %%

# %%
pred_0.shape
# %%
gt1.shape
gt1.shape
# %%
batch_len
# %%
trainer2.loss_fun(pred_0.unsqueeze(1),gt0.unsqueeze(1))
# %%
loss=torch.nn.MSELoss(reduction='mean')
# %%
loss(pred_0.unsqueeze(1),gt0.unsqueeze(1))
# %%
import torch
torch.round(trainer2.pred_0.sum(),3)
# %%
print(f'aaa{round(np.atleast_1d(trainer2.pred_0.sum().cpu().detach().numpy())[0],3)}')
# %%
np.float32(trainer2.pred_0.sum().cpu())
# %%
list(trainer2.pred_0.sum().cpu().detach().numpy())
# %%
list(trainer2.pred_0)
# %%
print(np.atleast_1d(trainer2.pred_0.sum().cpu().detach().numpy())[0])
# %%
trainer2.pred_0.sum()
# %%
print(f'aaaa   {round(np.atleast_1d(trainer2.pred_0.sum().cpu().detach().numpy())[0],3)}')
# %%
a='%.4f'%trainer2.pred_0.sum()
# %%
trainer2.model([v[0].cuda(),v[1].cuda()])
# %%
trainer2.pred_0.shape
# %%
trainer2.gt0.shape
# %%
trainer2.gt0
# %%
import torch
loss=torch.nn.MSELoss(reduction='mean')
loss(trainer2.pred_0,trainer2.gt0)
# %%
v[1].shape
# %%
trainer2.pred_.shape
# %%
trainer2.pred_[0].shape
# %%
for i,v in enumerate(loaders['val']):
    if i>13:
        break
print(v[1])

# %%
v[0]
# %%
import torch
model=trainer2.model
pred=model(v[0].type(torch.FloatTensor).cuda())
# %%
pred
# %%
