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
class TVLoss(nn.Module):
    def __init__(self, alpha):
        super(TVLoss, self).__init__()
        self.alpha = alpha
    def gradient(self,inp):
        # print(inp)
        r = F.pad(inp, (0, 1))[1:]
        r[-1]=inp[-1]
        # print(r)
        # l = F.pad(inp, (1, 0))[:-1]
        # l[0]=inp[0]
        # print(l)
        xgrad = torch.sum(torch.abs((r - inp)* 0.5))
        if xgrad==0:
            return xgrad
        else:
            return xgrad-0.5
        # print(xgrad)
        return xgrad
    def forward(self, y_true, y_pred):
        '''
        y_true:B*Length 0/1/-1
        y_pred:B*Length*class_n'''
        y_pred=y_pred.argmax(axis=2)
        tv_loss=0
        image_n=0
        for mt in range(y_pred.shape[0]):
            inp=torch.tensor([y_pred[mt][i] for i in range(y_pred.shape[1]) if y_true[mt][i]!=-1])
            tv_loss += self.gradient(inp) * self.alpha
            image_n+=inp.shape[0]
        # 计算损失
        
        # loss = torch.mean(torch.abs(y_true - y_pred)) * self.weight
        return tv_loss/image_n

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()
        
    def guassian_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """计算Gram核矩阵
        source: sample_size_1 * feature_size 的数据
        target: sample_size_2 * feature_size 的数据
        kernel_mul: 这个概念不太清楚, 感觉也是为了计算每个核的bandwith
        kernel_num: 表示的是多核的数量
        fix_sigma: 表示是否使用固定的标准差
            return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                            矩阵，表达形式:
                            [   K_ss K_st
                                K_ts K_tt ]
        """
        self.device = (torch.device('cuda')
            if source.is_cuda
            else torch.device('cpu'))
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0) # 合并在一起

        total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                        int(total.size(0)), \
                                        int(total.size(1)))
        #n个向量，每个向量与其他n-1个向量相减（n*n*l），每个元素求平方，在l维度上求和，得到距离矩阵（n*n）
        L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        # print(bandwidth_list)

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                    bandwidth_temp in bandwidth_list]
        # for i in kernel_val:
        #     print(i.shape)

        return sum(kernel_val) # 将多个核合并在一起

    def forward(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = int(source.size()[0])
        m = int(target.size()[0])

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        # print(kernels.shape)
        XX = kernels[:n, :n] 
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]
        # print(XX.shape)
        # print(YY.shape)
        XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
        XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

        YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
        YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss
class SupConLoss(nn.Module):
    
    def __init__(self, T=0.5):
        super(SupConLoss, self).__init__()
        self.T = T


    def forward(self, features, label):
        self.device = (torch.device('cuda')
            if features.is_cuda
            else torch.device('cpu'))
        T = self.T  #温度参数T
        # label = torch.tensor([1,0,1,0,1])
        n = label.shape[0]  # batch

        #这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask

        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = torch.ones(n ,n) - torch.eye(n, n )

        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/T)

        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0.to(self.device)


        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix


        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim


        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1)

        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum)


        loss = mask_no_sim + loss + torch.eye(n, n ).to(self.device)


        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  #求-log
        loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2naa

        # print(loss)  #0.9821
        #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
        # loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

        return loss
    
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
        if 'train' in self.dataloaders.keys():
            self.train_length=len(self.dataloaders['train'])
        else:
            self.train_length=len(self.dataloaders['source'])
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

    def save_gt_and_pred(self,PR_neme='gt_and_pred.pth'):
        PR_path=os.path.join(self.checkpoint_dir,PR_neme)
        print('SVAE FIGURE:',PR_path)
        self.metric.save_gt_and_pred(PR_path)

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
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=self.cfg['train']['gamma'])
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.cfg['train']['lr_policy'])
        return scheduler

    def collect_running_batch_states(self):
        running_acc=self.update_metric()[self.cfg['train']['acc_index']]
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est=self.timer_update()
        pred=torch.argmax(self.pred,dim=1)
        if self.is_training:
            if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()}    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                self.logger.write(message)
        else:
            if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()}\n'
                self.logger.write(message)

    def collect_epoch_states(self):
        scores=self.metric.get_scores()
        self.epoch_acc=scores[self.acc_index]
        self.logger.write(f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   epoch_{self.acc_index}: {self.epoch_acc} \n')
        message=''
        for k, v in scores.items():
            message+='%s: %.5f \n'%(k,v)
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
        self.model.load_state_dict(w2,False)
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

class Umultitime_Trainer(googlemultitime_Trainer):
    def __init__(self,dataloaders):
        super(Umultitime_Trainer,self).__init__(dataloaders)
    def collect_running_batch_states(self):
        running_acc=self.update_metric()[self.cfg['train']['acc_index']]
        m=len(self.dataloaders['train'])
        if not self.is_training:
            m=len(self.dataloaders['val'])
        est=self.timer_update()
        pred=torch.argmax(self.pred,dim=1)
        if self.is_training:
            if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                if self.cfg['train']['tv_loss']>0:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()} = {self.TVloss.item()} + {self.CEloss.item()} running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                else:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()} = {self.CEloss.item()} running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                
                self.logger.write(message)
        else:
            if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()}\n'
                self.logger.write(message)

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
                self.pred_=self.model(img)[0]
                
                if self.cfg['train']['tv_loss']>0:
                    self.TVloss_fun=TVLoss(self.cfg['train']['tv_loss'])
                    self.TVloss=self.TVloss_fun(self.batch[2],self.pred_)
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
                    self.CEloss=self.loss_fun(self.pred,self.gt,
                                            weight=w,
                                            ignore_index=self.cfg['train']['loss_fun_args']['ignore_index'])
                    if torch.isnan(self.CEloss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                if self.cfg['train']['tv_loss']>0:
                    self.loss=self.CEloss+self.TVloss
                else:

                    self.loss=self.CEloss
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
                    pred_=self.model(img)[0]
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
                    pred_=self.model(img)[0]
                    pred=pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                    self.pred=pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class']).detach()
                    self.batch_len=self.pred.shape[0]
                    # raise
            self.collect_running_batch_states()
        self.collect_epoch_states()

class UmultitimeMMD_Trainer(Umultitime_Trainer):
    def __init__(self,dataloaders):
        super(UmultitimeMMD_Trainer,self).__init__(dataloaders)
    def collect_running_batch_states(self):
        running_acc=self.update_metric()[self.cfg['train']['acc_index']]
        if 'target_labeled' in self.dataloaders.keys():  
            m=len(self.dataloaders['source'])
        else:
            m=len(self.dataloaders['source'])
        if not self.is_training:
            m=len(self.dataloaders['target_unlabeled'])
        est=self.timer_update()
        pred=torch.argmax(self.pred,dim=1)
        if self.is_training:
            if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                if 'target_labeled' in self.dataloaders.keys():
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()} =  {self.MMDloss.item()} + {self.SCloss.item()} + {self.CEloss.item()} + {self.TVloss.item()} running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                else:
                    message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h   loss: {self.loss.item()} =   + {self.CEloss.item()} + {self.TVloss.item()} running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()} \n'
                self.logger.write(message)
        else:
            if np.mod(self.batch_id,self.cfg['train']['print_step_interval'])==1:
                message=f'Is_training: {self.is_training}   epoch: {self.epoch_id}/{self.max_epochs-1}   batch: {self.batch_id}/{m}   need_time: {est}h    running_{self.acc_index}: {running_acc} sum: {pred.sum()}/{self.gt.sum()}\n'
                self.logger.write(message)

    def label_select(self,feature,label):
        label_flatten=label.reshape(-1).to(self.device)
        indices=torch.tensor([i for i in range(label_flatten.shape[0]) if label_flatten[i]>=0]).to(self.device)
        label_use=torch.index_select(label_flatten, 0, indices)
        v_len=feature.shape[-1]
        feature_flatten=feature.reshape((-1,v_len))
        feature_use=torch.index_select(feature_flatten, 0, indices)
        return feature_use, label_use
    def label_select_class(self,feature,label):
        output=[]
        for c in [0,1]:
            indices=torch.tensor([i for i in range(label.shape[0]) if label[i]==c]).to(self.device)
            # label_use=torch.index_select(label, 0, indices)
            feature_use=torch.index_select(feature, 0, indices)
            output.append(feature_use)
        return output
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
            
            
            
            self.tu_loader=iter(self.dataloaders['target_unlabeled'])
            if 'target_labeled' in self.dataloaders.keys():
                self.tl_loader=iter(self.dataloaders['target_labeled'])
            for self.batch_id,self.s_data in enumerate(self.dataloaders['source'],0):
                try:
                    self.tu_data= next(self.tu_loader)
                except StopIteration:
                    # print('a')
                    self.tu_loader = iter(self.dataloaders['target_unlabeled'])
                    self.tu_data= next(self.tu_loader)
                if 'target_labeled' in self.dataloaders.keys():    
                    try:
                        self.tl_data= next(self.tl_loader)
                    except StopIteration:
                        # print('b')
                        self.tl_loader = iter(self.dataloaders['target_labeled'])
                        self.tl_data= next(self.tl_loader)
                
                    self.s_len,self.tu_len,self.tl_len=self.s_data[0].shape[0],self.tu_data[0].shape[0],self.tl_data[0].shape[0]
                    self.batch=[]
                    for i in range(3):
                        self.batch.append(torch.cat([self.s_data[i],self.tu_data[i],self.tl_data[i]]))
                else:
                    self.s_len,self.tu_len=self.s_data[0].shape[0],self.tu_data[0].shape[0]
                    self.batch=[]
                    for i in range(3):
                        self.batch.append(torch.cat([self.s_data[i],self.tu_data[i]]))
        
        
                img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                    self.batch[1].type(torch.FloatTensor).to(self.device)]

                if torch.isnan(img[0]).any() or torch.isnan(img[1]).any():
                    print(';Exist NAN')
                    raise()

                ''''改'''
                # gt_=self.batch[2].flatten()#向下取整
                # gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                # self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long()
            
                
                self.pred_=self.model(img)
                self.feature=self.pred_[1].contiguous().detach().permute(0,2,1)
                if 'target_labeled' in self.dataloaders.keys():  
                    self.s_feature,self.tu_feature,self.tl_feature=self.feature[:self.s_len],self.feature[self.s_len:self.s_len+self.tu_len],self.feature[self.s_len+self.tu_len:]
                    self.s_label,self.tu_label,self.tl_label=self.batch[2][:self.s_len],self.batch[2][self.s_len:self.s_len+self.tu_len],self.batch[2][self.s_len+self.tu_len:]
                    self.tl_feature,self.tl_label=self.label_select(self.tl_feature,self.tl_label)
                else:
                    self.s_feature,self.tu_feature=self.feature[:self.s_len],self.feature[self.s_len:]
                    self.s_label,self.tu_label=self.batch[2][:self.s_len],self.batch[2][self.s_len:]
                    
                self.s_feature,self.s_label=self.label_select(self. s_feature,self.s_label)
                self.tu_feature,self.tu_label=self.label_select(self.tu_feature,self.tu_label)
                
                
                self.mmdloss_fun=MMDLoss()
                self.TVloss_fun=TVLoss(1)
                if 'target_labeled' in self.dataloaders.keys():
                    self.MMDloss=self.mmdloss_fun(self.s_feature,torch.cat([self.tu_feature,self.tl_feature]))
                    
                    self.class_feature=self.label_select_class(self.s_feature.contiguous(),self.s_label.contiguous())+\
                                       self.label_select_class(self.tl_feature.contiguous(),self.tl_label.contiguous())
                    self.class_label=[0,1,0,1]
                    self.prototypes=torch.stack([torch.mean(i,dim=0) for i in self.class_feature if i.shape[0]!=0])
                    self.prototypes_label=torch.tensor([self.class_label[i] for i in range(4) if self.class_feature[i].shape[0]!=0]).to(self.device)

                    self.SCloss_fun=SupConLoss()
                    self.SCloss=self.SCloss_fun(self.prototypes,self.prototypes_label)
                    
                    
                    
                    self.TVloss=self.TVloss_fun(
                        torch.cat([self.batch[2][:self.s_len],self.batch[2][self.s_len+self.tu_len:]]),
                        torch.cat([self.pred_[0][:self.s_len],self.pred_[0][self.s_len+self.tu_len:]])
                    )
                    
                    self.pred,self.gt=self.label_select(
                        torch.cat([self.pred_[0][:self.s_len],self.pred_[0][self.s_len+self.tu_len:]]),
                        torch.cat([self.batch[2][:self.s_len],self.batch[2][self.s_len+self.tu_len:]])
                    )
                

                else:
                    # self.MMDloss=self.mmdloss_fun(self.s_feature.contiguous(),self.tu_feature.contiguous())
                    self.TVloss=self.TVloss_fun(self.batch[2][:self.s_len],self.pred_[0][:self.s_len])
                    self.pred,self.gt=self.label_select(self.pred_[0][:self.s_len],self.batch[2][:self.s_len])

                    
                
                
                
                
                
                
                

                # self.pred=self.pred_.contiguous().view(-1,self.cfg['model']['args']['n_class'])
                # self.pred=self.pred[gt_mask!=-1].view(-1,self.cfg['model']['args']['n_class'])
                self.batch_len=self.pred.shape[0]
                # print(self.pred.argmax(axis=1))
                # print(self.batch[1])
                self.optimizer.zero_grad()

                if self.cfg['train']['loss']=='cross_entropy':
                    if self.cfg['train']['loss_fun_args']['weight'] is None:
                        w=None
                    else:
                        w=torch.tensor(self.cfg['train']['loss_fun_args']['weight']).to(torch.float32).to(self.device)
                    self.CEloss=self.loss_fun(self.pred,self.gt,
                                            weight=w,
                                            ignore_index=self.cfg['train']['loss_fun_args']['ignore_index'])
                    if torch.isnan(self.CEloss):
                        print(self.pred)
                        raise()
                else:
                    raise ValueError("Does not support binary_cross_entropy!")
                
                
                if 'target_labeled' in self.dataloaders.keys():
                    self.loss=self.MMDloss+self.SCloss+self.CEloss+self.TVloss
                else:
                    # self.MMDloss=0
                    self.loss=self.MMDloss+self.CEloss+self.TVloss
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

            for self.batch_id, self.batch in enumerate(self.dataloaders['target_unlabeled'], 0):
                with torch.no_grad():
                    img=[self.batch[0].type(torch.FloatTensor).to(self.device),
                        self.batch[1].type(torch.FloatTensor).to(self.device)]
                    gt_=self.batch[2].flatten()#向下取整
                    gt_mask=torch.cat([gt_.unsqueeze(1),gt_.unsqueeze(1)],dim=1)
                    self.gt=torch.tensor([i for i in list(gt_) if i !=-1]).to(self.device).long()
                    pred_=self.model(img)
                    pred=pred_[0].contiguous().view(-1,self.cfg['model']['args']['n_class'])
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