# Privacy and Data Enhancement

The official implementation of "On the Privacy Effect of Data Enhancement via the Lens of Memorization". In this work, we reproduced the LiRA method (Carlini et al. Membership Inference Attacks From First Principles) as the main privacy evaluation method for each data enhacement method.

### 1. Dependencies
- CUDA 11.0
- You can install the required packages by running: ```pip install -r requirements.txt```

### 2. Datasets
- CIFAR-10, CIFAR-100, and SVHN can be downloaded directly from torchvision.datasets.
- Purchase and Locations can be downloaded from https://github.com/privacytrustlab/datasets


### 3. File Structure
```
├── README.md
├── advtrain.py     # Functions used for adversarial training.
├── configs/        # Configuration files for training on different datasets.
├── dataset.py      # Functions used for loading datasets.
├── eval_privacy.py # Functions used for evaluating privacy using LiRA.
├── inference.py    # Functions used for computing $\phi$ used in LiRA.
├── models/         # DNN structures.
├── requirements.txt
├── sampleinfo/     # member and non-member information of 128 models used in LiRA.
├── trades_awp.py   # Functions used for AWP and TRADES-AWP training.
├── train.py        # Functions used for training (shadow) models.
├── utils.py        # Other functions.
└── utils_h.py      # Other functions.
```



### 4. Usage

This repository contains the code for training shadow models and performing LiRA. We support 12 data enhancement methods: "base", "smooth", "disturblabel", "noise", "cutout", "mixup", "jitter", "pgdat", "trades", "distillation", "AWP", "TradesAWP". The following steps are the instructions for reproducing the results in the paper. On CIFAR-10, we take one data augmentation method, Cutout, as an example:

Train the 128 shadow models for Cutout:
```
python train.py --train --s_model 0 --t_model 128 --aug_type cutout --dataset cifar10
```


Compute $\phi$ (required by LiRA) for each data point with multiple queries (Ensuring all 128 models trained):
```
python inference.py --mode eval --load_model --save_results --dataset cifar10 --query_mode multiple --aug_type cutout
```

Perform LiRA attacck after all $\phi$ computed (multiple queries version):
```
python eval_privacy.py --save_results --multi
```


### 5. Statement

If you find our work helpful for you, please consider to cite:
```
@article{li2022privacy,
  title={On the Privacy Effect of Data Enhancement via the Lens of Memorization},
  author={Xiao Li and Qiongxiu Li and Zhanhao Hu and Xiaolin Hu},
  journal={arXiv preprint arXiv:2208.08270},
  year={2022}
}
```
