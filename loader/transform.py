import torchvision.transforms.functional as tf
from torchvision import utils,transforms
import random
import numpy as np
def get_transform(trf_cfg):
    transform=[]
    for key in trf_cfg.keys():

        if key=='to_tensor' and trf_cfg['to_tensor']:
            transform.append(transforms.ToTensor())
        if key=='resize':
            transform.append(transforms.Resize((trf_cfg['resize'][0], trf_cfg['resize'][1])))
        if key=='randomcrop':
            transform.append(transforms.RandomCrop(trf_cfg['randomcrop'][0], padding=trf_cfg['randomcrop'][1]))
        if key=='normal':
            transform.append(transforms.Normalize((trf_cfg['normal'][0], trf_cfg['normal'][1])))
        if key=='h_flip' and trf_cfg['h_flip']:
            transform.append(transforms.RandomHorizontalFlip())
        if key=='v_flip' and trf_cfg['v_flip']:
            transform.append(transforms.RandomVerticalFlip())
    return transforms.Compose(transform)
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