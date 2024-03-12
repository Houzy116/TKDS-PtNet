#%%
import sys
import loader
import run
import warnings
import numpy as np
warnings.filterwarnings("ignore")
loaders=loader.get_loader()
trainer=run.get_trainer()
trainer=trainer(loaders)
trainer.train_models()
