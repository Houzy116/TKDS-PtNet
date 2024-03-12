#%%
import os
import numpy as np
import config.config_dict


'''
dataname格式:
    name_arg1-xxx_arg2-xxx_arg3.......
    xxx必须为数字
'''
class DataConfig():
    def __init__(self):
        self.root_dir='/home/hk/python_script/Ukraine/data/sample/'
        self.name=config.config_dict.cfg['dataset']['name']

        self.args=config.config_dict.cfg['dataset']['args']
        self.info={}
        
        #例如'sentinel2_size-6','sentinel2_size-4','sentinel2_size-2':

        eval('self.'+self.name)(**self.args)


    def sentinel2(self,size):
        size=int(size)
        self.info['train_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed/train/')
        self.info['test_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed/val/')
    
    def sentinel2littleN(self,size):
        size=int(size)
        self.info['train_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_littleN/train/')
        self.info['test_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_littleN/val/')

    def sentinel2greater0(self,size):
        size=int(size)
        self.info['train_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_greater_0/train/')
        self.info['test_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_greater_0/val/')
    
    def sentinel2greater1(self,size):
        size=int(size)
        self.info['train_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_greater_1/train/')
        self.info['test_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_greater_1/val/')

    def sentinel2class(self,size):
        size=int(size)
        self.info['train_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_class/train/')
        self.info['test_dir']=os.path.join(self.root_dir+'sample_sentinel2/',f'Livoberezhyny_size{size}_Destroyed_class/val/')


    def googledamagedsize6(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6/val.pth'
        
    def googledamagedsize6int(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_int/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_int/val.pth'

    def googledamagedsize6area(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA/val.pth'

    def googledamagedsize6areaint(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA_int/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA_int/val.pth'

    def googledamagedsize6areaAleppo(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA_int/cities/Aleppo/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA_int/cities/Aleppo/val.pth'

    def googledamagedsize6areaHama(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA/cities/Hama/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA/cities/Hama/val.pth'

    def googledamagedsize6areaHama120(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA/cities/Hama/train_size120.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_size6_AREA/cities/Hama/val_size120.pth'

    def googledamagedsize6areaHamaenhance(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_enhance_Hama_size6_AREA/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_enhance_Hama_size6_AREA/val.pth'

    def googledamagedHamaenhance(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_enhance_Hama/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_enhance_Hama/val.pth'

    def googledamagedsize6areaHamaall(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_all_Hama_size6_AREA/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_all_Hama_size6_AREA/val.pth'

    def googledamagedHamaall(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_all_Hama/train.pth'
        self.info['test_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_all_Hama/val.pth'
    
    def googledamagedsize6areaHamaallval(self,size):
        self.info['train_dir']='/data2/Syria_img/Syria_samples/split_havedamaged_all_Hama_size6_AREA/train.pth'
        self.info['test_dir']=f'/data2/Syria_img/Syria_samples/split_havedamaged_all_Hama_size6_AREA/val_{size}.pth'
    
    def googledamaged(self,size,city,mode):
        if mode is None:
            if city=='allcities':
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}/val.pth'
            else:
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}/cities/{city}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}/cities/{city}/val.pth'
        else:
            if city=='allcities':
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}/val.pth'
            else:
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}/cities/{city}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}/cities/{city}/val.pth'
    def googledamagedv2(self,size,city,mode):
        # city='Raqqa'
        if mode is None:
            if city=='allcities':
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v2/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v2/val.pth'
            else:
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v2/cities/{city}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v2/cities/{city}/val.pth'
        else:
            if city=='allcities':
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v2/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v2/val.pth'
            else:
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v2/cities/{city}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v2/cities/{city}/val.pth'
    def googledamagedv3(self,size,city,mode):
        # city='Hama'
        if mode is None:
            if city=='allcities':
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v3/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v3/val.pth'
            else:
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v3/cities/{city}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_v3/cities/{city}/val.pth'
        else:
            if city=='allcities':
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v3/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v3/val.pth'
            else:
                self.info['train_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v3/cities/{city}/train.pth'
                self.info['test_dir']=f'/ssd/hk/Syria_samples/split_havedamaged_size{size}_{mode}_v3/cities/{city}/val.pth'
    def Udamagedv2(self,size,city,mode):
        self.info['train_dir']=f'/mnt/nvme1n1/hk/Ukraine_samples/split_coordv2_v2/cities/{city}/train.pth'
        self.info['test_dir']=f'/mnt/nvme1n1/hk/Ukraine_samples/split_coordv2_v2/cities/{city}/val.pth'
    def Udamagedv3crossval(self,size,city,mode):
        self.info['train_dir']=f'/mnt/nvme1n1/hk/Ukraine_samples/split_coordv2_v3/cities/{city}/cross_val/{mode}/train.pth'
        self.info['test_dir']=f'/mnt/nvme1n1/hk/Ukraine_samples/split_coordv2_v3/cities/{city}/cross_val/{mode}/val.pth'
    def Udamagedv3(self,size,city,mode):
        self.info['train_dir']=f'/mnt/nvme1n1/hk/Ukraine_samples/split_coordv2_v3/cities/{city}/train.pth'
        self.info['test_dir']=f'/mnt/nvme1n1/hk/Ukraine_samples/split_coordv2_v3/cities/{city}/val.pth'

# %%
