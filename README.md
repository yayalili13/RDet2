# R(Det) 2: Randomized Decision Routing for Object Detection
[CVPR 2022] R (Det)^2: Randomized Decision Routing for Object Detection

# Introduction
This work is about applying soft decision trees into deep neural networks in an end-to-end learning manner for object detection. By combining soft decision trees, the decision choices and prediction values are disentangled. To facilitate the effective learning, the randomized decision routing is proposed with node selective and associative losses, which can boost the feature representative learning and network decision simultaneously. Experiments conducted on MS-COCO dataset demonstrate that R(Det)^2 is effective to improve the detection performance. 

This is a Python implementation, which is based on mmdetection. The installation, training and testing follows the mmdetection platform {https://github.com/open-mmlab/mmdetection}.

The code modification is on two aspects.

1) We add two files named as **node_selective_loss.py** and **node_associative_loss.py** under the file folder **mmdet/models/losses**;

2) We add one file names as **ra_roi_head.py** under the file folder **mmdet/models/roi_heads/** and one file names as **ra_convfc_bbox_head.py** under the file folder **mmdet/models/roi_heads/bbox_heads/**;

The config file is under the file folder **configs/rdet2/**. The **bbox_head** can also be easily plugged into other two-stage detectors.

This code is only released for academic use.

## Citation
>> @InProceedings{Li_2022_CVPR, \
>>    author    = {Li, Yali and Wang, Shengjin}, \
>>    title     = {R(Det)2: Randomized Decision Routing for Object Detection}, \
>>    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, \
>>    month     = {June}, \
>>    year      = {2022}, \
>>    pages     = {4825-4834} \
>> } 
