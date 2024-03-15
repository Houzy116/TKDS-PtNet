## TKDS-PtNet




TKDS-PtNet is a tool designed for monitoring damaged buildings. 
This repository includes a model comparison accuracy curve plotted by `make_figure.ipynb`. 
Additionally, it contains records of different model outputs and ground truth file paths. 

<p align="center">
  <img src="https://github.com/Houzy116/TKDS-PtNet/assets/131630519/aee80b34-ec59-4b26-acf3-22be6eabd2b6" alt="S7" width="600">
   <br>
  <em><strong>Fig.1 The comparison between the commonly used urban change detection scheme and TKDS.</strong> The detector can be PtNet, ResNet, or any other detection methods. </strong>a.</strong> Generation of the image patch samples. </strong>b.</strong> The commonly used destruction detection methods. </strong>c.</strong> The proposed TKDS scheme.</em>
</p>

  
### Usage

To train and validate the model, you can run `main.py`.
Configuration of training data sets, models, loaders, and various hyperparameters can be defined by modifying `config/config_dict.py`.

### Contents

- `make_figure.ipynb`: Jupyter notebook for plotting accuracy curves comparing different models.
- `main.py`: Script for training and validation.
- `config/config_dict.py`: Configuration file for defining datasets, models, loaders, and hyperparameters.
- `checkpoint/`: Directory containing log files from experiments mentioned in the paper.
- `data/sample`: Sample data used for training and validation.
- `data/fixed-effects`: Data and executable files for validating accuracy using fixed-effects models.
  
### Logging Details
The `checkpoint/` directory contains log files from experiments conducted with different configurations. These log files include detailed information about each experiment's `config_dict`, facilitating easy replication and comparison of results.
<p align="center">
  <img src="https://github.com/Houzy116/TKDS-PtNet/assets/131630519/ab501ac7-dd0b-4b15-87e1-b4d73b334a21" alt="S7" width="600">
   <br>
  <em><strong>Fig.2 The TKDS-PtNet architecture.</strong></em>
</p>


<p align="center">
  <img src="https://github.com/Houzy116/TKDS-PtNet/assets/131630519/35c7113b-7a9f-4b72-95e4-68aca8e25ab4" alt="S7" width="600">
   <br>
  <em><strong>Fig.3 The semi-supervised domain adaptation strategy for building damage detection, incorporating supervised contrastive learning and Maximum Mean Discrepancy.</strong></em>
</p>



### How to Use

1. Clone this repository to your local machine.
2. Install the necessary dependencies.
3. Run `main.py` to train and validate the model.
4. Modify `config/config_dict.py` to customize the training process according to your requirements.
5. Refer to `make_figure.ipynb` for visualizing and comparing the accuracy curves of different models.

### Citation

If you find TKDS-PtNet useful in your research, please consider citing:

