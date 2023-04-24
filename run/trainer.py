#%%
from re import L
import sys
sys.path.append('..')
import config
import numpy as np
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import model
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from run.metric_tool import ConfuseMatrixMeter
from run.metric_tool_seg import ConfuseMatrixMeterSEG
from run.logger_tool import Logger,Timer
import torch.nn.functional as F
import random
from collections import OrderedDict

# %%
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


class sentinel2_Trainer():
    def __init__(self,dataloaders):
        self.cfg=config.cfg
        self.gpu_ids=self.cfg['train']['gpu_ids']
        self.dataloaders=dataloaders
        self.model=model.get_model()
        self.use_gpus()
        self.init_weights()
        self.start_epochs=0
        self.max_epochs=self.cfg['train']['max_epochs']
        self.timer=Timer()
        self.train_length=len(self.dataloaders['train'])
        self.total_steps = (self.max_epochs - self.start_epochs)*self.train_length
        self.checkpoint_dir=self.cfg['root_dir']+'checkpoints/'+self.cfg['project_name']             
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        log_path = os.path.join(self.checkpoint_dir,'log.txt')
        self.logger = Logger(log_path)
        self.logger.write_dict_str(self.cfg)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg['train']['lr'],
                                momentum=self.cfg['train']['momentum'],
                                weight_decay=self.cfg['train']['w_decay'])
        self.lr_scheduler = self.get_scheduler()
        self.metric = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
        self.batch_len=0



        self.get_loss()
        self.load_acc()
        self.batch_id = 0
        self.epoch_id = 0
        self.is_training = False
        self.pred = None
        self.batch = None
        self.loss = None
        self.batch_id = 0
        self.epoch_id = 0        
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.acc_index=self.cfg['train']['acc_index']
        self.best_result_after_n=0



    def load_acc(self):
        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))
    
    def update_training_acc_curve(self):
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)
    
    def update_val_acc_curve(self):
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)
    
    def get_loss(self):
        if self.cfg['train']['loss']=='cross_entropy':
            self.loss_fun=F.cross_entropy
        elif self.cfg['train']['loss']=='binary_cross_entropy':
            self.loss_fun=F.binary_cross_entropy
        else:
            raise NotImplemented(self.cfg['train']['loss'])

    def timer_update(self):
        
        self.global_step=(self.epoch_id-self.start_epochs)*self.train_length+self.batch_id
        self.timer.update_progress((self.global_step+1)/self.total_steps)
        est=self.timer.estimated_remaining()#预计还需要多久完成
        # imps=(self.global_step+1)*self.cfg['loader']['batch_size']/self.timer.get_stage_elapsed()
        return est

    def use_gpus(self):
        gpu_ids_code=''
        for i in self.gpu_ids:
            gpu_ids_code+=str(i)
            gpu_ids_code+=','
        gpu_ids_code=gpu_ids_code[:-1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_code
        print(gpu_ids_code)
        self.device = torch.device("cuda:0"  if torch.cuda.is_available() and len(self.cfg['train']['gpu_ids'])>0
                    else "cpu")
        self.model.to(self.device)
        if len(self.gpu_ids) > 1:
            self.model= nn.DataParallel(self.model,device_ids=self.gpu_ids)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.model.apply(_init_weights)

    def get_scheduler(self):
        if self.cfg['train']['lr_policy'] == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - epoch / float(self.cfg['train']['max_epochs'] + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        elif self.cfg['train']['lr_policy'] == 'step':
            step_size = self.cfg['train']['step_size']
            gamma = self.cfg['train']['gamma']
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.cfg['train']['lr_policy'])
        return scheduler

    def collect_running_batch_states(self,write=True):
        running_acc=self.update_metric()[self.cfg['train']['acc_index']]
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est=self.timer_update()
        pred=torch.argmax(self.pred,dim=1)
        if write:
            if self.is_training:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()}    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                    self.logger.write(message)
            else:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()}\n'
                    self.logger.write(message)

    def collect_epoch_states(self,write=True):
        scores=self.metric.get_scores()
        self.epoch_acc=scores[self.acc_index]

        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_{self.acc_index}: {self.epoch_acc} \n')
        message=''
        for k, v in scores.items():
            message+='%s: %.5f \n'%(k,v)
        if write:    
            self.logger.write(message+'\n')
            self.logger.write('\n')

    def save_checkpoint(self,ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.lr_scheduler.state_dict(),
            }, os.path.join(self.checkpoint_dir, ckpt_name))

    def update_checkpoints(self):
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. Epoch_{self.acc_index}={self.epoch_acc}, Historical_best_{self.acc_index}={self.best_val_acc} (at epoch {self.best_epoch_id}\n)')
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc=self.epoch_acc
            self.best_epoch_id=self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*'*15+'best model updated!'+'*'*15+'\n')
            self.logger.write('\n')
            self.best_result_after_n=0
        self.best_result_after_n+=1
        if self.cfg['train']['loadbestmodel_whenbackwateracc'][0]:
            if self.best_result_after_n>self.cfg['train']['loadbestmodel_whenbackwateracc'][1]:

                self.load_checkpoint()
                print('Best model has not been updated for too long,load best model')
            
    def update_metric(self):
        target=self.gt.to(self.device).detach()
        pred=self.pred.detach()

        current_score=self.metric.update_cm(pr=pred.cpu(),gt=target.cpu())
        return current_score

    def load_best_w(self,ckpt_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir,ckpt_name)):
            self.logger.write('loading best checkpoint...\n')
            checkpoint=torch.load(os.path.join(self.checkpoint_dir,ckpt_name),
                                map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])
            # self.lr_scheduler.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])
            self.model.to(self.device)
            # self.start_epochs = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.total_steps = (self.max_epochs - self.start_epochs)*self.train_length
            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.start_epochs, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
            self.best_result_after_n=0
        else:
            print('training from scratch...')
    def load_checkpoint(self,ckpt_name='last_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir,ckpt_name)):
            self.logger.write('loading best checkpoint...\n')
            checkpoint=torch.load(os.path.join(self.checkpoint_dir,ckpt_name),
                                map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])
            self.model.to(self.device)
            self.start_epochs = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']
            self.total_steps = (self.max_epochs - self.start_epochs)*self.train_length
            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.start_epochs, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
            self.best_result_after_n=0
        else:
            print('training from scratch...')
    def PR_figure(self,PR_neme='bestmodel_PR_curve.jpg'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.PR_figure(PR_path)
    def load_pretrain_w(self):
        pretrain_w_file=os.path.join(self.cfg['train']['load_pretrain'][3],self.cfg['train']['load_pretrain'][2])
        pretrain_w_label=self.cfg['train']['load_pretrain'][1]
        print(f'Load pre-train weight:{pretrain_w_label}')
        print(f'from file:{pretrain_w_file}')
        w=torch.load(pretrain_w_file)
        w2={}
        if len(self.cfg['train']['gpu_ids'])==1:
            for k in w['model'].keys():
                if 'encoder.' in k:
                    k2=k.replace('encoder.module.','')
                    w2[k2]=w['model'][k]
        else:
            for k in w['model'].keys():
                if 'encoder.' in k:
                    k2=k.replace('encoder.','')
                    w2[k2]=w['model'][k]
        w2=OrderedDict(w2)
        self.model.load_state_dict(w2)
    def AUC_figure(self,AUC_neme='bestmodel_AUC_curve.jpg'):
        AUC_path=os.path.join(self.checkpoint_dir,AUC_neme)
        print('SVAE FIGURE:',AUC_path)
        self.metric.AUC_figure(AUC_path)
        
    def train_models(self):
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img10=self.batch[0].to(self.device)
                img20=self.batch[1].to(self.device)
                if torch.isnan(img10).any() or torch.isnan(img20).any():
                    print(';Exist NAN')
                    raise()
                self.pred=self.model([img10,img20])
                self.optimizer.zero_grad()
                self.gt=self.batch[2].to(self.device).long()#向下取整
                self.batch_len=self.pred.shape[0]
                if self.cfg['train']['loss']=='cross_entropy':
                    if self.cfg['train']['loss_fun_args']['weight'] is None:
                        w=None
                    else:
                        w=torch.tensor(self.cfg['train']['loss_fun_args']['weight']).to(torch.float32).to(self.device)
                    self.loss=self.loss_fun(self.pred,self.gt,
                                            weight=w,
                                            ignore_index=self.cfg['train']['loss_fun_args']['ignore_index'])
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()

            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img10=self.batch[0].to(self.device)
                    img20=self.batch[1].to(self.device)
                    self.pred=self.model([img10,img20])
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                img10=self.batch[0].to(self.device)
                img20=self.batch[1].to(self.device)
                self.pred=self.model([img10,img20])
                self.batch_len=self.pred.shape[0]
            self.collect_running_batch_states()
        self.collect_epoch_states()

class google_Trainer(sentinel2_Trainer):
    def __init__(self,dataloaders):
        super(google_Trainer,self).__init__(dataloaders)
    def update_metric(self):
        target=self.batch[1].to(self.device).detach()
        pred=self.pred.detach()
        current_score=self.metric.update_cm(pr=pred.cpu(),gt=target.cpu())
        return current_score
    def train_models(self):
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=self.batch[0].type(torch.FloatTensor).to(self.device)

                if torch.isnan(img).any():
                    print(';Exist NAN')
                    raise()
                self.pred=self.model(img)
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()
                self.gt=self.batch[1].to(self.device).long()#向下取整
                self.batch_len=self.pred.shape[0]
                if self.cfg['train']['loss']=='cross_entropy':
                    if self.cfg['train']['loss_fun_args']['weight'] is None:
                        w=None
                    else:
                        w=torch.tensor(self.cfg['train']['loss_fun_args']['weight']).to(torch.float32).to(self.device)
                    self.loss=self.loss_fun(self.pred,self.gt,
                                            weight=w,
                                            ignore_index=self.cfg['train']['loss_fun_args']['ignore_index'])
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()

            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self.gt=self.batch[1].to(self.device).long()
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    self.pred=self.model(img)
                    self.batch_len=self.pred.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                self.gt=self.batch[1].to(self.device).long()
                img=self.batch[0].type(torch.FloatTensor).to(self.device)
                self.pred=self.model(img)
                self.batch_len=self.pred.shape[0]
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)

class googlemultitime_Trainer(sentinel2_Trainer):
    def __init__(self,dataloaders):
        super(googlemultitime_Trainer,self).__init__(dataloaders)
    def load_pretrain_w(self):
        pretrain_w_file=os.path.join(self.cfg['train']['load_pretrain'][3],self.cfg['train']['load_pretrain'][2])
        pretrain_w_label=self.cfg['train']['load_pretrain'][1]
        print(f'Load pre-train weight:{pretrain_w_label}')
        print(f'from file:{pretrain_w_file}')
        w=torch.load(pretrain_w_file)
        w2={}
        if len(self.cfg['train']['gpu_ids'])==1:
            for k in w['model'].keys():
                if 'head' not in k:
                    k2=k.replace('module.','')
                    w2[k2]=w['model'][k]
        else:
            for k in w['model'].keys():
                if 'head' not in k:
                    w2[k]=w['model'][k]

        w2=OrderedDict(w2)
        self.model.load_state_dict(w2,False)
    def update_metric(self):
        target=self.gt.to(self.device).detach()
        pred=self.pred.detach()
        # print(pred)
        # print(torch.argmax(pred.cpu(),dim=1).numpy())
        current_score=self.metric.update_cm(pr=pred.cpu(),gt=target.cpu())
        return current_score
    def train_models(self,rebuilding_loader=None):
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=self.batch[0].type(torch.FloatTensor).to(self.device)

                if torch.isnan(img).any():
                    print(';Exist NAN')
                    raise()
                gt_=self.batch[1].flatten()#向下取整
                gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long()
                self.pred_=self.model(img)
                self.pred=self.pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                self.pred=self.pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class'])
                self.batch_len=self.pred.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                if self.cfg['train']['loss']=='cross_entropy':
                    if self.cfg['train']['loss_fun_args']['weight'] is None:
                        w=None
                    else:
                        w=torch.tensor(self.cfg['train']['loss_fun_args']['weight']).to(torch.float32).to(self.device)
                    self.loss=self.loss_fun(self.pred,self.gt,
                                            weight=w,
                                            ignore_index=self.cfg['train']['loss_fun_args']['ignore_index'])
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()
            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    gt_=self.batch[1].flatten()#向下取整
                    gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long()
                    pred_=self.model(img)
                    pred=pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                    self.pred=pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class'])
                    self.batch_len=self.pred.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()
            cities=['Aleppo','Hama','Homs','Raqqa','Deir-ez-Zor','Idlib']
            if self.cfg['v_generalization'][0]:
                for i in range(6):
                    city=cities[i]
                    if city==self.cfg['dataset']['args']['city']:
                        continue
                    self.logger.write(f'{city}\n')
                    self.metric.clear()
                    self.is_training = False
                    self.model.eval()
                    for self.batch_id, self.batch in enumerate(self.dataloaders['g'][i], 0):
                        with torch.no_grad():
                            img=self.batch[0].type(torch.FloatTensor).to(self.device)
                            gt_=self.batch[1].flatten()#向下取整
                            gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                            self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long()
                            pred_=self.model(img)
                            pred=pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                            self.pred=pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class'])
                            self.batch_len=self.pred.shape[0]
                        self.collect_running_batch_states(write=False)
                    self.collect_epoch_states(write=False)

                    # self.update_val_acc_curve()

            if rebuilding_loader is not None:
                self.test_rebuilding(rebuilding_loader,print_result=False,load_checkpoint=False)
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    gt_=self.batch[1].flatten()#向下取整
                    gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long().detach()
                    pred_=self.model(img)
                    pred=pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                    self.pred=pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class']).detach()
                    self.batch_len=self.pred.shape[0]
                    # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def test_rebuilding(self,rebuilding_loader,print_result=True,load_checkpoint=True):
        if load_checkpoint:
            seed_torch()
            self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation rebuilding...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(rebuilding_loader, 0):
            with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    gt_=self.batch[1].flatten()#向下取整
                    gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long().detach()
                    pred_=self.model(img)
                    pred=pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                    self.pred=pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class']).detach()
                    self.batch_len=self.pred.shape[0]
                    # raise
                    if print_result:
                        g=np.array(gt_.cpu()).reshape((-1,16))
                        # print(g.shape)
                        gg=[list(g[i,:]) for i in range(g.shape[0])]
                        p=np.array(torch.argmax(pred,dim=1).cpu()).reshape((-1,16))
                        pp=[list(p[i,:]) for i in range(p.shape[0])]
                        for i in range(g.shape[0]):
                            gg[i]=[gg[i][j] for j in range(len(gg[i])) if gg[i][j]!=-1]
                            pp[i]=[pp[i][j] for j in range(len(gg[i])) if gg[i][j]!=-1]
                        print('gt:')
                        print(gg)
                        print('pre:')
                        print(pp)
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)
class U_Trainer(sentinel2_Trainer):
    def __init__(self,dataloaders):
        super(U_Trainer,self).__init__(dataloaders)
    def update_metric(self):
        target=self.batch[2].to(self.device).detach()
        pred=self.pred.detach()
        # print(pred,target)
        # raise()
        current_score=self.metric.update_cm(pr=pred.cpu(),gt=target.cpu())
        return current_score
    def train_models(self):
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]

                if torch.isnan(img[0]).any() or torch.isnan(img[1]).any():
                    print(';Exist NAN')
                    raise()
                self.pred=self.model(img)
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()
                self.gt=self.batch[2].to(self.device).long()#向下取整
                self.batch_len=self.pred.shape[0]
                if self.cfg['train']['loss']=='cross_entropy':
                    if self.cfg['train']['loss_fun_args']['weight'] is None:
                        w=None
                    else:
                        w=torch.tensor(self.cfg['train']['loss_fun_args']['weight']).to(torch.float32).to(self.device)
                    self.loss=self.loss_fun(self.pred,self.gt,
                                            weight=w,
                                            ignore_index=self.cfg['train']['loss_fun_args']['ignore_index'])
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()

            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self.gt=self.batch[2].to(self.device).long()
                    img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                        self.batch[1].type(torch.FloatTensor).to(self.device)]
                    self.pred=self.model(img)
                    self.batch_len=self.pred.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                self.gt=self.batch[2].to(self.device).long()
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]
                self.pred=self.model(img)
                self.batch_len=self.pred.shape[0]
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)
class Umultitime_Trainer(googlemultitime_Trainer):
    def __init__(self,dataloaders):
        super(Umultitime_Trainer,self).__init__(dataloaders)
    def train_models(self):
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]

                if torch.isnan(img[0]).any() or torch.isnan(img[1]).any():
                    print(';Exist NAN')
                    raise()


                gt_=self.batch[2].flatten()#向下取整
                gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long()
                self.pred_=self.model(img)
                self.pred=self.pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                self.pred=self.pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class'])
                self.batch_len=self.pred.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                if self.cfg['train']['loss']=='cross_entropy':
                    if self.cfg['train']['loss_fun_args']['weight'] is None:
                        w=None
                    else:
                        w=torch.tensor(self.cfg['train']['loss_fun_args']['weight']).to(torch.float32).to(self.device)
                    self.loss=self.loss_fun(self.pred,self.gt,
                                            weight=w,
                                            ignore_index=self.cfg['train']['loss_fun_args']['ignore_index'])
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()

            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                        self.batch[1].type(torch.FloatTensor).to(self.device)]
                    gt_=self.batch[2].flatten()#向下取整
                    gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long()
                    pred_=self.model(img)
                    pred=pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                    self.pred=pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class'])
                    self.batch_len=self.pred.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                    img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                        self.batch[1].type(torch.FloatTensor).to(self.device)]
                    gt_=self.batch[2].flatten()#向下取整
                    gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long().detach()
                    pred_=self.model(img)
                    pred=pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                    self.pred=pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class']).detach()
                    self.batch_len=self.pred.shape[0]
                    # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)
class USAmultitime_Trainer(google_Trainer):
    def __init__(self,dataloaders):
        super(USAmultitime_Trainer,self).__init__(dataloaders)
        self.acc_index='R2'
    def train_models(self,rebuilding_loader=None):
        self.loss_fun=torch.nn.MSELoss(reduction='mean')
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=self.batch[0].type(torch.FloatTensor).to(self.device)

                if torch.isnan(img).any():
                    print(';Exist NAN')
                    raise()
                gt_=self.batch[1].flatten()#向下取整
                gt_mask=gt_.unsqueeze(1)
                self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                self.pred_=self.model(img)
                self.pred=self.pred_.contiguous().view(-1,1)
                self.pred=self.pred[gt_mask!=-1].view(-1,1).flatten()
                self.batch_len=self.pred.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                if self.cfg['train']['loss']=='cross_entropy':
                    self.loss=self.loss_fun(self.pred,self.gt)
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()
            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    gt_=self.batch[1].flatten()#向下取整
                    gt_mask=gt_.unsqueeze(1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                    pred=self.model(img)
                    pred=pred.contiguous().view(-1,1)
                    self.pred=pred[gt_mask!=-1].view(-1,1).flatten()
                    self.batch_len=self.pred.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()

                    # self.update_val_acc_curve()

            if rebuilding_loader is not None:
                self.test_rebuilding(rebuilding_loader,print_result=False,load_checkpoint=False)
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    gt_=self.batch[1].flatten()#向下取整
                    gt_mask=gt_.unsqueeze(1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long().detach()
                    pred_=self.model(img)
                    pred=pred_.contiguous().view(-1,1)
                    self.pred=pred[gt_mask!=-1].view(-1,1).detach()
                    self.batch_len=self.pred.shape[0]
                    # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def collect_running_batch_states(self,write=True):
        running_acc=self.update_metric()['R2']
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est=self.timer_update()
        pred=self.pred
        if write:
            if self.is_training:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()}    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                    self.logger.write(message)
            else:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()}\n'
                    self.logger.write(message)

    def collect_epoch_states(self,write=True):
        scores=self.metric.get_scores2()
        self.epoch_acc=scores[self.acc_index]

        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_{self.acc_index}: {self.epoch_acc} \n')
        message=''
        for k, v in scores.items():
            message+='%s: %.5f \n'%(k,v)
        if write:    
            self.logger.write(message+'\n')
            self.logger.write('\n')
    def update_metric(self):
        target=self.gt.to(self.device).detach()
        pred=self.pred.detach()
        # print(pred)
        # print(torch.argmax(pred.cpu(),dim=1).numpy())
        current_score=self.metric.update_cm2(pr=pred.cpu(),gt=target.cpu())
        return current_score
    def update_checkpoints(self):
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. Epoch_{self.acc_index}={self.epoch_acc}, Historical_best_{self.acc_index}={self.best_val_acc} (at epoch {self.best_epoch_id}\n)')
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc=self.epoch_acc
            self.best_epoch_id=self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*'*15+'best model updated!'+'*'*15+'\n')
            self.logger.write('\n')
            self.best_result_after_n=0
        self.best_result_after_n+=1
        if self.cfg['train']['loadbestmodel_whenbackwateracc'][0]:
            if self.best_result_after_n>self.cfg['train']['loadbestmodel_whenbackwateracc'][1]:

                self.load_checkpoint()
                print('Best model has not been updated for too long,load best model')
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)
class USA_Trainer(USAmultitime_Trainer):
    def __init__(self,dataloaders):
        super(USA_Trainer,self).__init__(dataloaders)
        self.acc_index='R2'
    def train_models(self,rebuilding_loader=None):
        self.loss_fun=torch.nn.MSELoss(reduction='mean')
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=self.batch[0].type(torch.FloatTensor).to(self.device)

                if torch.isnan(img).any():
                    print(';Exist NAN')
                    raise()
                self.gt=self.batch[1].type(torch.FloatTensor).div(14400).flatten().to(self.device)#向下取整
                # gt_mask=gt_.unsqueeze(1)
                # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                self.pred_=self.model(img)
                self.pred=self.pred_.contiguous().view(-1,1).flatten()
                # self.pred=self.pred[gt_mask!=-1].view(-1,1)
                self.batch_len=self.pred.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                if self.cfg['train']['loss']=='cross_entropy':
                    self.loss=self.loss_fun(self.pred,self.gt)
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()
            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    self.gt=self.batch[1].type(torch.FloatTensor).div(14400).flatten().to(self.device)#向下取整

                    # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                    self.pred_=self.model(img)
                    self.pred=self.pred_.contiguous().view(-1,1).flatten()
                    self.batch_len=self.pred.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()

                    # self.update_val_acc_curve()

            if rebuilding_loader is not None:
                self.test_rebuilding(rebuilding_loader,print_result=False,load_checkpoint=False)
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    self.gt=self.batch[1].type(torch.FloatTensor).div(14400).flatten().to(self.device)#向下取整

                    # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                    self.pred_=self.model(img)
                    self.pred=self.pred_.contiguous().view(-1,1).flatten()
                    self.batch_len=self.pred.shape[0]
                    # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def collect_running_batch_states(self,write=True):
        running_acc=self.update_metric()['R2']
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est=self.timer_update()
        pred=self.pred
        if write:
            if self.is_training:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()}    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                    self.logger.write(message)
            else:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()}\n'
                    self.logger.write(message)
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)
    def collect_epoch_states(self,write=True):
        scores=self.metric.get_scores2()
        self.epoch_acc=scores[self.acc_index]

        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_{self.acc_index}: {self.epoch_acc} \n')
        message=''
        for k, v in scores.items():
            message+='%s: %.5f \n'%(k,v)
        if write:    
            self.logger.write(message+'\n')
            self.logger.write('\n')
    def update_metric(self):
        target=self.gt.to(self.device).detach()
        pred=self.pred.detach()
        # print(pred)
        # print(torch.argmax(pred.cpu(),dim=1).numpy())
        current_score=self.metric.update_cm2(pr=pred.cpu(),gt=target.cpu())
        return current_score
    def update_checkpoints(self):
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. Epoch_{self.acc_index}={self.epoch_acc}, Historical_best_{self.acc_index}={self.best_val_acc} (at epoch {self.best_epoch_id}\n)')
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc=self.epoch_acc
            self.best_epoch_id=self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*'*15+'best model updated!'+'*'*15+'\n')
            self.logger.write('\n')
            self.best_result_after_n=0
        self.best_result_after_n+=1
        if self.cfg['train']['loadbestmodel_whenbackwateracc'][0]:
            if self.best_result_after_n>self.cfg['train']['loadbestmodel_whenbackwateracc'][1]:

                self.load_checkpoint()
                print('Best model has not been updated for too long,load best model')

class CENmultitime_Trainer(google_Trainer):
    def __init__(self,dataloaders):
        super(CENmultitime_Trainer,self).__init__(dataloaders)
        self.acc_index='R2'
        self.metric0 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
        self.metric1 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
        self.metric2 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
    def train_models(self,rebuilding_loader=None):
        self.loss_fun=torch.nn.MSELoss(reduction='mean')
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric0.clear()
            self.metric1.clear()
            self.metric2.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]
                if torch.isnan(img[0]).any() or torch.isnan(img[1]).any():
                    print(';Exist NAN')
                    raise()
                gt_all=self.batch[2]
                gt_0=gt_all[:,:,0].flatten()#向下取整
                gt_1=gt_all[:,:,1].flatten()
                gt_2=gt_all[:,:,2].flatten()
                self.gt0=torch.tensor([i for i in list(gt_0) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt1=torch.tensor([i for i in list(gt_1) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt2=torch.tensor([i for i in list(gt_2) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.pred_=self.model(img)
                self.pred_0=self.pred_[:,:,0].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.pred_1=self.pred_[:,:,1].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.pred_2=self.pred_[:,:,2].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.batch_len=self.pred_0.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                self.loss0=self.loss_fun(self.pred_0,self.gt0)
                self.loss1=self.loss_fun(self.pred_1,self.gt1)
                self.loss2=self.loss_fun(self.pred_2,self.gt2)
                self.loss=self.loss0+self.loss1+self.loss2
                if torch.isnan(self.loss):
                    print(self.pred_)
                    raise()
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric0.clear()
            self.metric1.clear()
            self.metric2.clear()
            self.is_training = False
            self.model.eval()
            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                        self.batch[1].type(torch.FloatTensor).to(self.device)]
                    if torch.isnan(img[0]).any() or torch.isnan(img[1]).any():
                        print(';Exist NAN')
                        raise()
                    gt_all=self.batch[2]
                    gt_0=gt_all[:,:,0].flatten()#向下取整
                    gt_1=gt_all[:,:,1].flatten()
                    gt_2=gt_all[:,:,2].flatten()
                    self.gt0=torch.tensor([i for i in list(gt_0) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.gt1=torch.tensor([i for i in list(gt_1) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.gt2=torch.tensor([i for i in list(gt_2) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.pred_=self.model(img)
                    self.pred_0=self.pred_[:,:,0].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                    self.pred_1=self.pred_[:,:,1].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.pred_2=self.pred_[:,:,2].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.batch_len=self.pred_0.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()

                    # self.update_val_acc_curve()

            if rebuilding_loader is not None:
                self.test_rebuilding(rebuilding_loader,print_result=False,load_checkpoint=False)
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric0.clear()
        self.metric1.clear()
        self.metric2.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]
                if torch.isnan(img[0]).any() or torch.isnan(img[1]).any():
                    print(';Exist NAN')
                    raise()
                gt_all=self.batch[2]
                gt_0=gt_all[:,:,0].flatten()#向下取整
                gt_1=gt_all[:,:,1].flatten()
                gt_2=gt_all[:,:,2].flatten()
                self.gt0=torch.tensor([i for i in list(gt_0) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt1=torch.tensor([i for i in list(gt_1) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt2=torch.tensor([i for i in list(gt_2) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.pred_=self.model(img)
                self.pred_0=self.pred_[:,:,0].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.pred_1=self.pred_[:,:,1].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.pred_2=self.pred_[:,:,2].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.batch_len=self.pred_0.shape[0]
                # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def collect_running_batch_states(self,write=True):
        running_acc0,running_acc1,running_acc2  =self.update_metric()
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est='%.3f'%self.timer_update()
        pred=[self.pred_0,self.pred_1,self.pred_2]
        gt=[self.gt0,self.gt1,self.gt2]
        if self.is_training:
            loss='%.3f'%self.loss.item()
        running_acc_str='%.3f'%running_acc0+'    '+'%.3f'%running_acc1+'    '+'%.3f'%running_acc2
        sum_str='%.3f'%pred[0].sum()+'/'+'%.3f'%gt[0].sum()+'    '
        sum_str+='%.3f'%pred[1].sum()+'/'+'%.3f'%gt[1].sum()+'    '
        sum_str+='%.3f'%pred[2].sum()+'/'+'%.3f'%gt[2].sum()
        if write:
            if self.is_training:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {loss}    running_{self.acc_index}: {running_acc_str} sum: {sum_str}\n'
                    self.logger.write(message)
            else:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc_str} sum: {sum_str}\n'
                    self.logger.write(message)

    def collect_epoch_states(self,write=True):
        scores0=self.metric0.get_scores2()
        scores1=self.metric1.get_scores2()
        scores2=self.metric2.get_scores2()
        self.epoch_acc0=scores0[self.acc_index]
        self.epoch_acc1=scores1[self.acc_index]
        self.epoch_acc2=scores2[self.acc_index]
        self.epoch_acc=(self.epoch_acc0+self.epoch_acc1+self.epoch_acc2)/3

        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_mR2: {self.epoch_acc} \n')
        message=''
        for k, v in scores0.items():
            message+='%s: %.5f \n'%(k,v)
        for k, v in scores1.items():
            message+='%s: %.5f \n'%(k,v)
        for k, v in scores2.items():
            message+='%s: %.5f \n'%(k,v)
        if write:    
            self.logger.write(message+'\n')
            self.logger.write('\n')
    def update_metric(self):
        target0=self.gt0.to(self.device).detach()
        target1=self.gt1.to(self.device).detach()
        target2=self.gt2.to(self.device).detach()
        pred_0=self.pred_0.detach()
        pred_1=self.pred_1.detach()
        pred_2=self.pred_2.detach()
        # print(pred)
        # print(torch.argmax(pred.cpu(),dim=1).numpy())
        current_score0=self.metric0.update_cm2(pr=pred_0.cpu(),gt=target0.cpu())
        current_score1=self.metric1.update_cm2(pr=pred_1.cpu(),gt=target1.cpu())
        current_score2=self.metric2.update_cm2(pr=pred_2.cpu(),gt=target2.cpu())
        return [current_score0['R2'],current_score1['R2'],current_score2['R2']]
    def update_checkpoints(self):
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. Epoch_{self.acc_index}={self.epoch_acc}, Historical_best_{self.acc_index}={self.best_val_acc} (at epoch {self.best_epoch_id}\n)')
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc=self.epoch_acc
            self.best_epoch_id=self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*'*15+'best model updated!'+'*'*15+'\n')
            self.logger.write('\n')
            self.best_result_after_n=0
        self.best_result_after_n+=1
        if self.cfg['train']['loadbestmodel_whenbackwateracc'][0]:
            if self.best_result_after_n>self.cfg['train']['loadbestmodel_whenbackwateracc'][1]:

                self.load_checkpoint()
                print('Best model has not been updated for too long,load best model')
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric0.save_gt_and_pred(PR_path.replace('.pth','_0.pth'))
        self.metric1.save_gt_and_pred(PR_path.replace('.pth','_1.pth'))
        self.metric2.save_gt_and_pred(PR_path.replace('.pth','_2.pth'))

class CEN_Trainer(CENmultitime_Trainer):
    def __init__(self,dataloaders):
        super(CEN_Trainer,self).__init__(dataloaders)
        self.acc_index='R2'
        self.metric0 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
        self.metric1 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
        self.metric2 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
    def train_models(self,rebuilding_loader=None):
        self.loss_fun=torch.nn.MSELoss(reduction='mean')
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        # checkpoint=torch.load('',map_location=self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric0.clear()
            self.metric1.clear()
            self.metric2.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]

                gt_all=self.batch[2]
                self.gt0=gt_all[:,0].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.gt1=gt_all[:,1].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt2=gt_all[:,2].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                # gt_mask=gt_.unsqueeze(1)
                # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                self.pred_=self.model(img)
                self.pred_0=self.pred_[:,0].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.pred_1=self.pred_[:,1].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.pred_2=self.pred_[:,2].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                # self.pred=self.pred_.contiguous().view(-1,1).flatten()
                # self.pred=self.pred[gt_mask!=-1].view(-1,1)
                self.batch_len=self.pred_0.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                self.loss0=self.loss_fun(self.pred_0,self.gt0)
                self.loss1=self.loss_fun(self.pred_1,self.gt1)
                self.loss2=self.loss_fun(self.pred_2,self.gt2)
                self.loss=self.loss0+self.loss1+self.loss2
                if torch.isnan(self.loss):
                    print(self.pred_)
                    raise()
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise
                # raise()
            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()

           
            self.logger.write('Begin evaluation...\n')
            self.metric0.clear()
            self.metric1.clear()
            self.metric2.clear()
            self.is_training = False
            self.model.eval()
            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                        self.batch[1].type(torch.FloatTensor).to(self.device)]

                    gt_all=self.batch[2]
                    self.gt0=gt_all[:,0].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                    self.gt1=gt_all[:,1].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.gt2=gt_all[:,2].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    # gt_mask=gt_.unsqueeze(1)
                    # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                    self.pred_=self.model(img)
                    self.pred_0=self.pred_[:,0].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                    self.pred_1=self.pred_[:,1].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.pred_2=self.pred_[:,2].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    # self.pred=self.pred_.contiguous().view(-1,1).flatten()
                    # self.pred=self.pred[gt_mask!=-1].view(-1,1)
                    self.batch_len=self.pred_0.shape[0]
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()

                    # self.update_val_acc_curve()

            if rebuilding_loader is not None:
                self.test_rebuilding(rebuilding_loader,print_result=False,load_checkpoint=False)
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric0.clear()
        self.metric1.clear()
        self.metric2.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]

                gt_all=self.batch[2]
                self.gt0=gt_all[:,0].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.gt1=gt_all[:,1].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt2=gt_all[:,2].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                # gt_mask=gt_.unsqueeze(1)
                # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                self.pred_=self.model(img)
                self.pred_0=self.pred_[:,0].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.pred_1=self.pred_[:,1].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.pred_2=self.pred_[:,2].flatten().type(torch.FloatTensor).to(self.device).unsqueeze(1)
                # self.pred=self.pred_.contiguous().view(-1,1).flatten()
                # self.pred=self.pred[gt_mask!=-1].view(-1,1)
                self.batch_len=self.pred_0.shape[0]
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def collect_running_batch_states(self,write=True):
        running_acc0,running_acc1,running_acc2  =self.update_metric()
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est='%.3f'%self.timer_update()
        pred=[self.pred_0,self.pred_1,self.pred_2]
        gt=[self.gt0,self.gt1,self.gt2]
        loss='%.3f'%self.loss.item()
        running_acc_str='%.3f'%running_acc0+'    '+'%.3f'%running_acc1+'    '+'%.3f'%running_acc2
        sum_str='%.3f'%pred[0].sum()+'/'+'%.3f'%gt[0].sum()+'    '
        sum_str+='%.3f'%pred[1].sum()+'/'+'%.3f'%gt[1].sum()+'    '
        sum_str+='%.3f'%pred[2].sum()+'/'+'%.3f'%gt[2].sum()
        if write:
            if self.is_training:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {loss}    running_{self.acc_index}: {running_acc_str} sum: {sum_str}\n'
                    self.logger.write(message)
            else:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc_str} sum: {sum_str}\n'
                    self.logger.write(message)

    def collect_epoch_states(self,write=True):
        scores0=self.metric0.get_scores2()
        scores1=self.metric1.get_scores2()
        scores2=self.metric2.get_scores2()
        self.epoch_acc0=scores0[self.acc_index]
        self.epoch_acc1=scores1[self.acc_index]
        self.epoch_acc2=scores2[self.acc_index]
        self.epoch_acc=(self.epoch_acc0+self.epoch_acc1+self.epoch_acc2)/3

        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_mR2: {self.epoch_acc} \n')
        message=''
        for k, v in scores0.items():
            message+='%s: %.5f \n'%(k,v)
        for k, v in scores1.items():
            message+='%s: %.5f \n'%(k,v)
        for k, v in scores2.items():
            message+='%s: %.5f \n'%(k,v)
        if write:    
            self.logger.write(message+'\n')
            self.logger.write('\n')
    def update_metric(self):
        target0=self.gt0.to(self.device).detach()
        target1=self.gt1.to(self.device).detach()
        target2=self.gt2.to(self.device).detach()
        pred_0=self.pred_0.detach()
        pred_1=self.pred_1.detach()
        pred_2=self.pred_2.detach()
        # print(pred)
        # print(torch.argmax(pred.cpu(),dim=1).numpy())
        current_score0=self.metric0.update_cm2(pr=pred_0.cpu(),gt=target0.cpu())
        current_score1=self.metric1.update_cm2(pr=pred_1.cpu(),gt=target1.cpu())
        current_score2=self.metric2.update_cm2(pr=pred_2.cpu(),gt=target2.cpu())
        return [current_score0['R2'],current_score1['R2'],current_score2['R2']]
    def update_checkpoints(self):
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. Epoch_{self.acc_index}={self.epoch_acc}, Historical_best_{self.acc_index}={self.best_val_acc} (at epoch {self.best_epoch_id}\n)')
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc=self.epoch_acc
            self.best_epoch_id=self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*'*15+'best model updated!'+'*'*15+'\n')
            self.logger.write('\n')
            self.best_result_after_n=0
        self.best_result_after_n+=1
        if self.cfg['train']['loadbestmodel_whenbackwateracc'][0]:
            if self.best_result_after_n>self.cfg['train']['loadbestmodel_whenbackwateracc'][1]:

                self.load_checkpoint()
                print('Best model has not been updated for too long,load best model')
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric0.save_gt_and_pred(PR_path.replace('.pth','_0.pth'))
        self.metric1.save_gt_and_pred(PR_path.replace('.pth','_1.pth'))
        self.metric2.save_gt_and_pred(PR_path.replace('.pth','_2.pth'))

class USAmultitimev4_Trainer(google_Trainer):
    def __init__(self,dataloaders):
        super(USAmultitimev4_Trainer,self).__init__(dataloaders)
        self.acc_index='R2'
        self.metric0 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
        self.metric1 = ConfuseMatrixMeter(n_class=self.cfg['model']['args']['n_class'])
    def train_models(self,rebuilding_loader=None):
        self.loss_fun=torch.nn.MSELoss(reduction='mean')
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric0.clear()
            self.metric1.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=self.batch[0].type(torch.FloatTensor).to(self.device)

                if torch.isnan(img).any():
                    print(';Exist NAN')
                    raise()

                gt_all=self.batch[1]
                gt_0=gt_all[:,:,0].flatten()#向下取整
                gt_1=gt_all[:,:,1].flatten()
                self.gt0=torch.tensor([i for i in list(gt_0) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt1=torch.tensor([i for i in list(gt_1) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)  
                self.pred_=self.model(img)                              
                self.pred_0=self.pred_[:,:,0].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.pred_1=self.pred_[:,:,1].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.batch_len=self.pred_0.shape[0]


                # self.pred_=self.model(img)
                # self.pred=self.pred_.contiguous().view(-1,1)
                # self.pred=self.pred[gt_mask!=-1].view(-1,1).flatten()
                # self.batch_len=self.pred.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()
                self.loss0=self.loss_fun(self.pred_0,self.gt0)
                self.loss1=self.loss_fun(self.pred_1,self.gt1)
                self.loss=self.loss0+self.loss1
                if torch.isnan(self.loss):
                    print(self.pred_)
                    raise()

                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric0.clear()
            self.metric1.clear()
            self.is_training = False
            self.model.eval()
            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)

                    if torch.isnan(img).any():
                        print(';Exist NAN')
                        raise()

                    gt_all=self.batch[1]
                    gt_0=gt_all[:,:,0].flatten()#向下取整
                    gt_1=gt_all[:,:,1].flatten()
                    self.gt0=torch.tensor([i for i in list(gt_0) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.gt1=torch.tensor([i for i in list(gt_1) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)  
                    self.pred_=self.model(img)                              
                    self.pred_0=self.pred_[:,:,0].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                    self.pred_1=self.pred_[:,:,1].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                    self.batch_len=self.pred_0.shape[0]

                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()

                    # self.update_val_acc_curve()

            if rebuilding_loader is not None:
                self.test_rebuilding(rebuilding_loader,print_result=False,load_checkpoint=False)
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric0.clear()
        self.metric1.clear()
        self.metric2.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                img=self.batch[0].type(torch.FloatTensor).to(self.device)

                if torch.isnan(img).any():
                    print(';Exist NAN')
                    raise()

                gt_all=self.batch[1]
                gt_0=gt_all[:,:,0].flatten()#向下取整
                gt_1=gt_all[:,:,1].flatten()
                self.gt0=torch.tensor([i for i in list(gt_0) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.gt1=torch.tensor([i for i in list(gt_1) if i !=-1]).type(torch.FloatTensor).to(self.device).unsqueeze(1)  
                self.pred_=self.model(img)                              
                self.pred_0=self.pred_[:,:,0].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)#向下取整
                self.pred_1=self.pred_[:,:,1].flatten()[gt_0!=-1].type(torch.FloatTensor).to(self.device).unsqueeze(1)
                self.batch_len=self.pred_0.shape[0]
                # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()

    def collect_running_batch_states(self,write=True):
        running_acc0,running_acc1 =self.update_metric()
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est='%.3f'%self.timer_update()
        pred=[self.pred_0,self.pred_1]
        gt=[self.gt0,self.gt1]
        if self.is_training:
            loss='%.3f'%self.loss.item()
        running_acc_str='%.3f'%running_acc0+'    '+'%.3f'%running_acc1
        sum_str='%.3f'%pred[0].sum()+'/'+'%.3f'%gt[0].sum()+'    '
        sum_str+='%.3f'%pred[1].sum()+'/'+'%.3f'%gt[1].sum()
        if write:
            if self.is_training:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {loss}    running_{self.acc_index}: {running_acc_str} sum: {sum_str}\n'
                    self.logger.write(message)
            else:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc_str} sum: {sum_str}\n'
                    self.logger.write(message)

    def collect_epoch_states(self,write=True):
        scores0=self.metric0.get_scores2()
        scores1=self.metric1.get_scores2()
        self.epoch_acc0=scores0[self.acc_index]
        self.epoch_acc1=scores1[self.acc_index]
        self.epoch_acc=(self.epoch_acc0+self.epoch_acc1)/2

        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_mR2: {self.epoch_acc} \n')
        message=''
        for k, v in scores0.items():
            message+='%s: %.5f \n'%(k,v)
        for k, v in scores1.items():
            message+='%s: %.5f \n'%(k,v)
        if write:    
            self.logger.write(message+'\n')
            self.logger.write('\n')
    def update_metric(self):
        target0=self.gt0.to(self.device).detach()
        target1=self.gt1.to(self.device).detach()
        pred_0=self.pred_0.detach()
        pred_1=self.pred_1.detach()
        # print(pred)
        # print(torch.argmax(pred.cpu(),dim=1).numpy())
        current_score0=self.metric0.update_cm2(pr=pred_0.cpu(),gt=target0.cpu())
        current_score1=self.metric1.update_cm2(pr=pred_1.cpu(),gt=target1.cpu())
        return [current_score0['R2'],current_score1['R2']]
    def update_checkpoints(self):
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. Epoch_{self.acc_index}={self.epoch_acc}, Historical_best_{self.acc_index}={self.best_val_acc} (at epoch {self.best_epoch_id}\n)')
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc=self.epoch_acc
            self.best_epoch_id=self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*'*15+'best model updated!'+'*'*15+'\n')
            self.logger.write('\n')
            self.best_result_after_n=0
        self.best_result_after_n+=1
        if self.cfg['train']['loadbestmodel_whenbackwateracc'][0]:
            if self.best_result_after_n>self.cfg['train']['loadbestmodel_whenbackwateracc'][1]:

                self.load_checkpoint()
                print('Best model has not been updated for too long,load best model')
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric0.save_gt_and_pred(PR_path.replace('.pth','_0.pth'))
        self.metric1.save_gt_and_pred(PR_path.replace('.pth','_1.pth'))

class USASEG_Trainer(USAmultitime_Trainer):
    def __init__(self,dataloaders):
        super(USA_Trainer,self).__init__(dataloaders)
        self.acc_index='R2'
        self.metric = ConfuseMatrixMeterSEG(n_class=2)
    def train_models(self,rebuilding_loader=None):
        # self.loss_fun=torch.nn.MSELoss(reduction='mean')
        if self.cfg['train']['load_pretrain'][0]:
            self.load_pretrain_w()
        if self.cfg['train']['load_checkpoint']:
            self.load_checkpoint()
        seed_torch()
        for self.epoch_id in range(self.start_epochs,self.max_epochs):
            self.metric.clear()
            self.is_training=True
            self.model.train()
            self.logger.write('lr: %0.7f\n' % self.optimizer.param_groups[0]['lr'])
            for self.batch_id,self.batch in enumerate(self.dataloaders['train'],0):
                img=self.batch[0].type(torch.FloatTensor).to(self.device)
                self.gt=self.batch[1].type(torch.FloatTensor).to(self.device)

                # if torch.isnan(img).any():
                #     print(';Exist NAN')
                #     raise()
                # self.gt=self.batch[1].type(torch.FloatTensor).flatten().to(self.device)#向下取整
                self.pred_=self.model(img)
                # self.pred=self.pred_.contiguous().view(-1,1).flatten()
                # self.pred=self.pred[gt_mask!=-1].view(-1,1)
                # self.batch_len=self.pred.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                if self.cfg['train']['loss']=='cross_entropy':
                    self.loss=self.loss_fun(self.pred,self.gt)
                    if torch.isnan(self.loss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                self.loss.backward()
                self.optimizer.step()
                self.collect_running_batch_states()
                self.timer_update()
                self.lr_scheduler.step()
                # if self.batch_id>2:
                #     raise

            self.collect_epoch_states()
            self.update_training_acc_curve()
            # self.lr_scheduler.step()


            self.logger.write('Begin evaluation...\n')
            self.metric.clear()
            self.is_training = False
            self.model.eval()
            for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    img=self.batch[0].type(torch.FloatTensor).to(self.device)
                    self.gt=self.batch[1].type(torch.FloatTensor).to(self.device)
                    # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                    self.pred_=self.model(img)
                self.collect_running_batch_states()
            self.collect_epoch_states()
            self.update_val_acc_curve()
            self.update_checkpoints()

                    # self.update_val_acc_curve()

            if rebuilding_loader is not None:
                self.test_rebuilding(rebuilding_loader,print_result=False,load_checkpoint=False)
    def test_model(self):
        seed_torch()
        self.load_best_w(ckpt_name='best_ckpt.pt')
        self.logger.write('Begin evaluation...\n')
        self.metric.clear()
        self.is_training = False
        self.epoch_id=self.best_epoch_id
        self.model.eval()
        for self.batch_id, self.batch in enumerate(self.dataloaders['val'], 0):
            with torch.no_grad():
                img=self.batch[0].type(torch.FloatTensor).to(self.device)
                self.gt=self.batch[1].type(torch.FloatTensor).to(self.device)
                # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).type(torch.FloatTensor).div(14400).to(self.device)
                self.pred_=self.model(img)
                # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()
    def collect_running_batch_states(self,write=True):
        running_acc=self.update_metric()['miou']
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est=self.timer_update()
        pred=self.pred
        if write:
            if self.is_training:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()}    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                    self.logger.write(message)
            else:
                if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()}\n'
                    self.logger.write(message)
    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)
    def collect_epoch_states(self,write=True):
        scores=self.metric.get_scores2()
        self.epoch_acc=scores[self.acc_index]

        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_{self.acc_index}: {self.epoch_acc} \n')
        message=''
        for k, v in scores.items():
            message+='%s: %.5f \n'%(k,v)
        if write:    
            self.logger.write(message+'\n')
            self.logger.write('\n')
    def update_metric(self):
        target=self.gt.to(self.device).detach()
        pred=self.pred.detach()
        # print(pred)
        # print(torch.argmax(pred.cpu(),dim=1).numpy())
        current_score=self.metric.update_cm2(pr=pred.cpu(),gt=target.cpu())
        return current_score
    def update_checkpoints(self):
        self.save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write(f'Lastest model updated. Epoch_{self.acc_index}={self.epoch_acc}, Historical_best_{self.acc_index}={self.best_val_acc} (at epoch {self.best_epoch_id}\n)')
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc=self.epoch_acc
            self.best_epoch_id=self.epoch_id
            self.save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*'*15+'best model updated!'+'*'*15+'\n')
            self.logger.write('\n')
            self.best_result_after_n=0
        self.best_result_after_n+=1
        if self.cfg['train']['loadbestmodel_whenbackwateracc'][0]:
            if self.best_result_after_n>self.cfg['train']['loadbestmodel_whenbackwateracc'][1]:

                self.load_checkpoint()
                print('Best model has not been updated for too long,load best model')
    def get_scheduler(self):
        def get_lr(train_steps,warmup_steps=1500,warmup_ratio=1e-6,max_steps=15000,power=1):
            if warmup_steps and train_steps < warmup_steps:
                # warmup_percent_done = train_steps / warmup_steps
                warmup_learning_rate = warmup_ratio * train_steps  #gradual warmup_lr
                learning_rate = warmup_learning_rate
            else:
                #learning_rate = np.sin(learning_rate)  #预热学习率结束后,学习率呈sin衰减
                learning_rate = (warmup_steps*warmup_ratio) * (1-(train_steps-warmup_steps)/(mxa_steps-warmup_steps))**power
            if train_steps>max_steps:
                print("train_steps:%.3f--warmup_steps:%.3f--learning_rate:%.3f" % (
                        train_steps+1,warmup_steps,learning_rate))
            return learning_rate
        scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=get_lr)
        return scheduler