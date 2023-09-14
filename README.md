# Feature Alignment and Uniformity for Test Time Adaptation
This repo is officical PyTorch implement of Feature Alignment and Uniformity for Test Time Adaptation (**CVPR 2023**).  
This paper could be found at [arXiv](https://arxiv.org/abs/2303.10902), [open access](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Feature_Alignment_and_Uniformity_for_Test_Time_Adaptation_CVPR_2023_paper.html) and [IEEEXplore](https://ieeexplore.ieee.org/document/10203978).  
This codebase is mainly based on [T3A](https://github.com/matsuolab/T3A) and [DeepDG](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG).  
## 💻 Dependence 
```
torch
torchvision
numpy
tqdm
timm
sklearn
pillow
```
## Dataset
Download datasets used in our paper from:  
[PACS](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd)  
[OfficeHome](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC)  
[VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)  
[DomainNet](http://ai.bu.edu/M3SDA/)  
## Train source model
Please use `train.py` to train the source model. For example:
```
python train.py --dataset PACS --data_dir your_data_dir --opt_type Adam --lr 5e-5 --max_epoch 50
```
Change `--dataset PACS` for other datasets, such as `office-home`, `VLCS`, `DomainNet`.  
Set `--net` to use different backbones, such as `resnext50`, `ViT-B16`.
## Test time adaptation
```
python unsupervise_adapt.py --dataset PACS\
                            --data_dir your_data_dir\
                            --adapt_alg TSD\ 
                            --pretrain_dir your_pretrain_model_dir\
                            --lr 1e-4
```
Change `--adapt_alg TSD` to use different methods of test time adaptation, e.g. `T3A`, `SHOT-IM`, `Tent`.  
`--pretrain_dir` denotes the path of source model, e.g. `./train_outputs/model.pkl`.  
Empirically, set `--lr` to 1e-4 or 1e-5 achieves good performance.
You can also search it using _training domain validation set_.
## 📝 Citation
If this repo is useful for your research, please consider citing our paper:
```
@inproceedings{wang2023feature,
  title={Feature alignment and uniformity for test time adaptation},
  author={Wang, Shuai and Zhang, Daoan and Yan, Zipei and Zhang, Jianguo and Li, Rui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20050--20060},
  year={2023}
}
```
## ✉️ Contact
Please contact bit.ybws@gmail.com

