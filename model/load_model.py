#%%
import sys
sys.path.append('..')
import config
def get_model():
    cfg=config.cfg
    name=cfg['model']['name']
    args=cfg['model']['args']
    pack = __import__('model.'+name)
    model_module = getattr(pack, name)
    model_class = getattr(model_module, name)
    model=model_class(**args)
    return model