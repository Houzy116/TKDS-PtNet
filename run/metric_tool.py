from cgitb import grey
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from sklearn.metrics import precision_recall_curve,roc_curve,r2_score

import matplotlib.pyplot as plt
import os

###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.gt=[]
        self.pr=[]
        self.pred=None

    def initialize(self, val, weight,gt,pr,pred):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True
        self.gt=gt
        self.pr=pr
        self.pred=pred

    def update(self,val,gt,pr,pred,weight=1):

        gt=list(np.array(gt).flatten())
        pr=list(np.array(pr).flatten())
        if not self.initialized:
            self.initialize(val, weight,gt,pr,pred)
        else:
            self.add(val, weight,gt,pr,pred)

    def add(self, val, weight,gt,pr,pred):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
        self.gt+=gt
        self.pr+=pr
        self.pred=np.concatenate((self.pred,pred),axis=0)
        # print(self.pred.shape)

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum,self.gt,self.pr)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        pred=pr.detach().numpy()
        # print(pred)
        pr=torch.argmax(pr,dim=1).numpy()
        # print(pr)
        gt=gt.numpy()
        # print(gt)
        # raise()
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        
        self.update(val=val,gt=gt,pr=pr,pred=pred,weight=weight)
        
        mF1,F1= cm2F1(val)
        auc= roc_auc(pred,gt)
        return {'mF1':mF1,'auc':auc,'F1_0':F1[0],'F1_1':F1[1]}
    def update_cm2(self, pr, gt, weight=1):
        # print(pred)
        pr=pr.numpy()
        # print(pr)
        gt=gt.numpy()
        # print(gt)
        # raise()

        
        self.update(val=0,gt=gt,pr=pr,pred=pr,weight=weight)
        R2= r2_score(pr,gt)
        return {'R2':R2}

    def get_scores(self):
        scores_dict = cm2score(self.sum,self.gt,self.pr,self.pred)
        return scores_dict
    def get_scores2(self):
        scores_dict = {'R2':r2_score(self.gt,self.pr)}
        return scores_dict
    def PR_figure(self,PR_savename):
        PR_curve(self.pred,self.gt,PR_savename)
    
    def AUC_figure(self,AUC_savename):
        AUC_curve(self.pred,self.gt,AUC_savename)

    def save_gt_and_pred(self,AUC_savename):
        save_gt_and_pred_file(self.pred,self.gt,AUC_savename)
def roc_auc(pred,gt):
    try:
        pred=torch.Tensor(pred)
        pred=F.softmax(pred,dim=1)
        # print(gt)
        # print(pred)
        roc=roc_auc_score(gt, pred[:,1])
        return roc
    except:
        return 0.5
def PR_curve(pred,gt,PR_savename):

    # print(gt)
    # print(pred)
    pred=torch.Tensor(pred)
    pred=F.softmax(pred,dim=1)
    precision, recall, thresholds = precision_recall_curve(gt, pred[:,1])
    plt.figure('P-R Curve')
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall,precision)
    plt.savefig(PR_savename,dpi=600)
    plt.show()
    torch.save([gt, pred[:,1]],os.path.join(os.path.dirname(PR_savename),'gt_and_pred.pth'))
    # torch.save([gt, pred],os.path.join(os.path.dirname(PR_savename),'gt_and_pred.pth'))
def save_gt_and_pred_file(pred,gt,savename):
    torch.save([gt, pred],savename)
    # torch.save([gt, pred],os.path.join(os.path.dirname(PR_savename),'gt_and_pred.pth')) 
    
def AUC_curve(pred,gt,AUC_savename):
    pred=torch.Tensor(pred)
    pred=F.softmax(pred,dim=1)
    # print(gt)
    # print(pred)
    
    fpr, tpr, thresholds = roc_curve(gt, pred[:,1],pos_label=1)
    plt.figure('AUC-ROC Curve')
    plt.title('AUC/ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr,tpr)
    plt.savefig(AUC_savename,dpi=600)
    plt.show()

def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1,F1


def cm2score(confusion_matrix,gt,pr,pred):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))
    auc=roc_auc(pred,gt)
    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1,'auc':auc}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    # print(2,label_gts.shF
    for lt, lp in zip(label_gts, label_preds):

        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']