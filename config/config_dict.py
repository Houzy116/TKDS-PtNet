#%%
a={
    'size':6,
    'model':'mynet6mtc1v2',
    'country':'Syria',
    'city':'allcities',
    'mode':'all',
    'to120':False,
    'use_samples_percent':1,
    'addpre':[False,
            'mt'
            ],
    'multi_rebuiling':False,
    'multi_time':[True,
                'RandomDelandCopyv2',#None,'RandomDel','LabelMask','Shuffle','RandomDelandCopy','RandomDelandCopyv2'
                ],
    'v_generalization':[False,'channels3normalized'],#'allnormalized'
    'v':3, #数据集version
    'cross_val':[False,5],
    'pretrain':[False,
                'SupCon1000v2',
                'ckpt_epoch_1000.pth',
                '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/MSRV_models/SupCon_MSRV_Umynet6_lr_0.05_decay_0.0001_bsz_2048_temp_0.07_trial_0_cosine_warm'
                # '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/MSRV_models/SupCon_MSRV_Umynet6_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_0_cosine',
                # '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/U_size6_models/SupCon_U_size6_Umynet6_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_0_cosine',
                # '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/allcities_size120v2_models/SupCon_allcities_size120v2_myresnet18_lr_0.05_decay_0.0001_bsz_256_temp_0.07_trial_0_cosine',
                # '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/allcities_size6v2_models/SupCon_allcities_size6v2_mynet6_lr_0.05_decay_0.0001_bsz_1024_temp_0.07_trial_0_cosine_warm/',
                # '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/allcities_size120v2_models/SupCon_allcities_size120v2_mynet6size120_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_0_cosine/',
                # '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/HamaandRaqqa_size6v2_models/SupCon_HamaandRaqqa_size6v2_resnet9128_lr_0.05_decay_0.0001_bsz_1024_temp_0.07_trial_0_cosine_warm/'
                # '/home/hk/python_script/SupContrast-master/SupContrast-master/save/SupCon/HamaandRaqqa_size6v2_models/SupCon_HamaandRaqqa_size6v2_mynet6_lr_0.05_decay_0.0001_bsz_1024_temp_0.07_trial_0_cosine_warm/'
                ]}

model=a['model']
usp=a['use_samples_percent']
if a['size']==120:
    loader='google'
else:
    loader='googlesize6'

size=a['size']
country=a['country']
city=a['city']
mode=a['mode']
if a['to120']:
    if a['multi_time']:
        loader='googlesize6upresamplemultitime'
        size='6to120'
        a['size']=6
    else:
        loader='googlesize6upresample'
        size='6to120'
        a['size']=6
if a['multi_time'][0]:
    if a['size']==6:
    
        loader='googlesize6multitime'
        # a['v']=3
        loader_mode=a['multi_time'][1]
        m='googlemultitime'
    else:
        loader='googlesize120multitime'
        # a['v']=3
        loader_mode=a['multi_time'][1]
        m='googlemultitime'
elif a['addpre'][0]:
    if a['size']==6:
        loader='googleadd3presize6'
    else:
        loader='googleadd3pre'
    m='google'
    loader_mode=a['addpre'][1]
else:
    m='google'
    loader_mode=None
if mode is None:
    pn=f'试一试_{model}_{country}_size{str(size)}_{city}_effectweight'
else:
    pn=f'试一试_{model}_{country}_size{str(size)}_{city}{mode}_effectweight'
if a['v']==2:
    pn+='_v2'
    datasetname='googledamagedv2'
    # if country=='Ukrain':
    #     datasetname='Udamagedv2'
    #     loader='Usize6' 
    #     m='U'
    if a['model']=='Uresnet50' or a['model']=='Uresnet18':
        loader='Usize120' 
        m='google'

elif a['v']==3:
    pn+='_v3'
    datasetname='googledamagedv3' 
    if country=='Ukrain':
        if a['cross_val'][0]:
            mode=a['cross_val'][1]
            datasetname='Udamagedv3crossval'
        else:
            datasetname='Udamagedv3' 
        loader='Usize6multitime' 
        m='Umultitime'
else:
    datasetname='googledamaged'
    if country=='Ukrain':
        datasetname='Udamagedv2'
        m='U'
# loader='Usize6RGB'
# m='google'

if a['multi_rebuiling']:
    datasetname='googledamagedv3rebuilingponly'
    pn+='_rebuilingmultiponly'
if a['v_generalization'][0]:
    gen_method=a['v_generalization'][1]
    pn+=f'_g_{gen_method}'
# pn+='_4'
if city=='USA':
    if a['v']==3:
        if not a['multi_time'][0]:
            raise('multi_time is False')
        datasetname='USAv3'
        a['size']=120
        loader='USAsize120multitime'
        model='USAmynet6size120mtc1v2'
        loader_mode=None
        m='USAmultitime'
        pn=f'试一试_{model}_USA_v3'
    elif a['v']==4:
        print(4)
        if not a['multi_time'][0]:
            raise('multi_time is False')
        datasetname='USAv4'
        a['size']=120
        loader='USAsize120multitimev4'
        model='USAmynet6size120mtc1v2v4'
        loader_mode=None
        m='USAmultitimev4'
        pn=f'试一试_{model}_USA_v4'
    else:
        if a['multi_time'][0]:
            raise('multi_time is False')
        datasetname='USAv2'
        a['size']=120
        loader='USAsize120'
        model='USAresnet18'  
        loader_mode=None 
        m='USA'   
        pn=f'试一试_{model}_USA_v2'
elif city=='USA-SEG':
    if a['v']==2:
        if a['multi_time'][0]:
            raise('multi_time is False')
        datasetname='USASEGv2'
        a['size']=120
        loader='SEGsize120'
        model='myseg6size120'  
        loader_mode=None 
        m='USA'   
        pn=f'试一试_{model}_USASEG_v2'
    elif a['v']==3:
        if not a['multi_time'][0]:
            raise('multi_time is False')
        datasetname='USASEGv3'
        a['size']=120
        loader='SEGsize120multitime'
        model='myseg6size120mtc1v2'  
        loader_mode=None 
        m='USA'   
        pn=f'试一试_{model}_USASEG_v3'
elif city=='CEN':
    if a['v']==3:
        if not a['multi_time'][0]:
            raise('multi_time is False')
        datasetname='CENv3'#
        a['size']=120
        loader='CENsize120multitime'
        model='CENmynet6size120mtc1v2'
        loader_mode=None
        m='CENmultitime'
        pn=f'试一试_{model}_CEN_v3'
    else:
        if a['multi_time'][0]:
            raise('multi_time is False')
        datasetname='CENv2'
        a['size']=120
        loader='CENsize120'
        model='CENresnet18'  
        loader_mode=None 
        m='CEN'   
        pn=f'试一试_{model}_CEN_v2'
if a['to120']:
    if a['multi_time']:
        loader='googlesize6upresamplemultitime'
        size='6to120'
        a['size']=6
    else:
        loader='googlesize6upresample'
        size='6to120'
        a['size']=6
if usp!=1:
    pn+=f'_usesamples{int(usp*100)}percent'
if a['multi_time'][0]:
    pn+='_multi-time'
if a['pretrain'][0]:
    pn+='_'+a['pretrain'][1]
if loader_mode is not None:
    pn+='_'
    pn+=loader_mode
# pn+='_weight45_trainMSRV'
pn+='_2'
if a['cross_val'][0]:
    pn+=f'_weight45_crossval{mode}'
cfg={'project_name':pn,
     'root_dir':'/home/hk/python_script/Ukraine/',
    'dataset':{'name':datasetname,
                'args':{'size':a['size'],'city':a['city'],'mode':mode}},
    'transform_train':{
                'to_tensor':True,
                'hv_flip_together':True,
                # 'v_flip':True,
                # 'h_flip':True,
                # 'resize':[32,32],
                # 'randomcrop':[32,4],#size,padding
                # 'normal':[[],[]]
                },
    #val_transform
    'transform_val':{
                'to_tensor':True,
                'hv_flip_together':False,
                },
    'loader':{'name':loader,
                # 'name':'google',
                # 'name':'googlesize6',
                # 'name':'googlesize6upresample',
                'mode':loader_mode,
                'use_samples_percent':usp,
                'batch_size':64*16,#1024
                'num_workers':8,
                'shuffle':True
            },
    'model':{'name':model,
                # 'name':'simplenet',
                # 'name':'simplenetgoogle',
                # 'name':'resnet18',
                # 'name':'pnasnet',
                # 'name':'resnet18',
                # 'name':'resnet50',
                'args':{'n_class':2}      
            },
    'train':{
                'name':m,
                # 'name':'sentinel2',
                'gpu_ids':[0],
                'lr':0.0025,
                'lr_policy':'step',#or 'linear'
                'loadbestmodel_whenbackwateracc':[False,10],
                'step_size':15,
                'gamma':0.5,
                'momentum':0.5,
                'w_decay':5e-4,
                'max_epochs':135,
                'loss':'cross_entropy',# or binary_cross_entropy(暂时不支持)
                'load_pretrain':a['pretrain'],
                'load_checkpoint':True,
                #weitght:None or [1,9] or [x,x]  ignore_index=2 or -100(default) ignore_index不能用于binary_cross_entropy
                # [0.51979147, 13.13170347]Hama no enhance
                # 
                
                'loss_fun_args':{'weight':[1, 9],'ignore_index':-100},
                'print_step_interval':100,
                'acc_index':'F1_1'
            },
    'v_generalization':a['v_generalization']
                }
if a['multi_rebuiling']:
    cfg['train']['loss_fun_args']['weight']=[9, 1]
if a['multi_time'][0]:
    # print('aaa')
    # cfg['loader']['batch_size']=int(cfg['loader']['batch_size']/16)
    if city=='U':
        cfg['loader']['batch_size']=int(cfg['loader']['batch_size']/20)
    elif city=='USA' and a['v']>=3:
        cfg['loader']['batch_size']=int(cfg['loader']['batch_size']/12)
    elif city=='CEN' and a['v']==3:
        cfg['loader']['batch_size']=int(cfg['loader']['batch_size']/5)
    elif city=='USA-SEG' and a['v']==3:
        cfg['loader']['batch_size']=int(cfg['loader']['batch_size']/9)
    else:
        cfg['loader']['batch_size']=int(cfg['loader']['batch_size']/16)
# %%












