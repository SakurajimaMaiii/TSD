# TSD
PyTorch implement of Feature Alignment and Uniformity for Test Time Adaptation (CVPR 2023).  
[arXiv](https://arxiv.org/abs/2303.10902), [open access](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Feature_Alignment_and_Uniformity_for_Test_Time_Adaptation_CVPR_2023_paper.html).  
This codebase is mainly based on [T3A](https://github.com/matsuolab/T3A) and [DeepDG](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG).  
## Dependence 
```
torch
torchvision
numpy
tqdm
timm
sklearn
```
## Dataset
[PACS](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd)  
[OfficeHome](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC)  
[VLCS](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8)  
[DomainNet](http://ai.bu.edu/M3SDA/)  
## Train source model
Please use `train.py` to train the source model. For example:
```
python train.py --dataset PACS --data_dir your_data_dir 
```
You can set different datasets or networks.
## Test time adaptation
```
python unsupervise_adapt.py --dataset PACS \
                            --data_dir your_data_dir \
                            --adapt_alg TSD \ 
                            --pretrain_dir your_pretrain_model_dir
```
## Citation
```
@InProceedings{Wang_2023_CVPR,
    author    = {Wang, Shuai and Zhang, Daoan and Yan, Zipei and Zhang, Jianguo and Li, Rui},
    title     = {Feature Alignment and Uniformity for Test Time Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20050-20060}
}
```


