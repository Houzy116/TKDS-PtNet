#%%
import sys
# from Ukraine.model.load_model import get_model
import loader
# from python_script.Ukraine.loader.loader import get_rebuiling_loader
import run
import warnings
import numpy as np
warnings.filterwarnings("ignore")
loaders={'train':loader.get_rebuiling_loader('/ssd/hk/Ukraine_samples/split_coordv2_v3/cities/MSV/samples2.pth',batch_size=128),
        'val':loader.get_rebuiling_loader('/ssd/hk/Ukraine_samples/split_coordv2_v3/cities/Rubizhne/samples.pth',batch_size=128)}

trainer=run.get_trainer()
trainer2=trainer(loaders)
# n=trainer2.model()

# %%
# rebuilding_split_path='/ssd/hk/Syria_samples/split_havedamaged_size6_all_v3_rebuilding/val.pth'
# rebuilding_loader=loader.get_rebuiling_loader(rebuilding_split_path,batch_size=4)
# trainer.train_models(rebuilding_loader)
trainer2.train_models()
# trainer2.test_model()
# trainer2.save_gt_and_pred()
trainer2.PR_figure(PR_neme=f'bestmodel_PR_curve_val.jpg')
trainer2.AUC_figure(AUC_neme=f'bestmodel_AUC_curve_val.jpg')
# trainer2.PR_figure(PR_neme=f'bestmodel_PR_curve.jpg')
# trainer2.AUC_figure(AUC_neme=f'bestmodel_AUC_curve.jpg')
#%%
# p=f'/ssd/hk/Ukraine_samples/split_coordv2_v3/cities/Azovstal/train.pth'
# l=loader.get_rebuiling_loader(p,batch_size=64)
# trainer2.logger.write(f'++++++++++++++++++ val  Azovstal ++++++++++++++++++\n')
# loaders['val']=l
# trainer2=trainer(loaders)
# trainer2.test_model()