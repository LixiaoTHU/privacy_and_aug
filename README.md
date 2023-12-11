# Privacy, DA, and AT

We take one DA method, Cutout, as an example:

Train the 128 shadow models for Cutout:

    python train.py --train --s_model 0 --t_model 128 --aug_type cutout --dataset cifar10

Inference $\phi$ for each data point with multiple queries (Ensure all models trained):

    python inference.py --mode eval --load_model --save_results --dataset cifar10 --query_mode multiple --aug_type cutout

Perform LiRA after all $\phi$ computed (multiple queries):

    python eval_privacy.py --save_results --multi


If you find this project helpful for you, please consider cite:
```
@article{li2022privacy,
  title={On the Privacy Effect of Data Enhancement via the Lens of Memorization},
  author={Li, Xiao and Li, Qiongxiu and Hu, Zhanhao and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2208.08270},
  year={2022}
}
```
