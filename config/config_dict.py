cfg={'project_name':pn,
     'root_dir':'/home/ubuntu/python_script/Ukraine/',
    'dataset':{'name':'Udamagedv3',
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
    'loader':{
                'name':'Usize6multitime',
                # 'name':'google',
                # 'name':'googlesize6',
                # 'name':'googlesize6upresample',
                'mode':loader_mode,
                'use_samples_percent':usp,
                'batch_size':20*50,#1024
                'num_workers':8,
                'shuffle':False
            },
    'model':{'name':model,

                'args':{'n_class':2}      
            },
    'train':{
                'name':'Umultitime',
                # 'name':'sentinel2',
                'gpu_ids':[2],
                'lr':0.005,
                'lr_policy':'step',#or 'linear'
                'loadbestmodel_whenbackwateracc':[False,10],
                'step_size':30,
                'gamma':0.5,
                'momentum':0.9,
                'w_decay':5e-4,
                
                'max_epochs':30,
                'loss':'cross_entropy',# or binary_cross_entropy(暂时不支持)
                'tv_loss':1,
                'load_pretrain':a['pretrain'],
                'load_checkpoint':False,
                #weitght:None or [1,9] or [x,x]  ignore_index=2 or -100(default) ignore_index不能用于binary_cross_entropy
                # [0.51979147, 13.13170347]Hama no enhance
                # 
                
                'loss_fun_args':{'weight':[1, 9],'ignore_index':-100},
                'print_step_interval':50,
                'acc_index':'F1_1'
                # 'acc_index':'precision'
            },
    'v_generalization':a['v_generalization']
                }