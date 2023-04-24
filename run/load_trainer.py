#%%
import sys
sys.path.append('..')
import config
import run.trainer
def get_trainer():
    cfg=config.cfg
    name=cfg['train']['name']
    trainer_name=name+'_Trainer'
    trainer_class = getattr(run.trainer, trainer_name)
    return trainer_class
#%%
