# Panoramic Annular Semantic Segmentation
Panoramic Annular Semantic Segmentation in PyTorch

# PASS Dataset

## GoogleDrive
[**New! Google Drive Download Link**](https://drive.google.com/file/d/1A_P2u5HUbrHZnKJYAOL2f7JLxxj69LqB/view?usp=sharing)

## BaiduYun
For Validation (Most important files):

[**Unfolded Panoramas for Validation**](https://pan.baidu.com/s/1lsd_CN9u4uSCp-KmE2pn9Q),
(400 images)

[**Annonations**](https://pan.baidu.com/s/1XJ6fFq60UwTZui456AQlPw), (400 annotation images)

[Groundtruth](https://pan.baidu.com/s/1RkrxtYu5Y1UzBvzn8aBugg)

There are 400 panoramas with annotations. Please use the Annotations data for evaluation.

In total, there are 1050 panoramas. Complete Panoramas:

[All Unfolded Panoramas](https://pan.baidu.com/s/16BLZArMyVfP_dEYnshEicQ)

RAW Panoramas: [RAW1](https://pan.baidu.com/s/1LBTQnVHcL0TKoY7njtPiBg),
               [RAW2](https://pan.baidu.com/s/1B_kaC8uu531exuXMlCE6_A),
               [RAW3](https://pan.baidu.com/s/1car_7_dH58wKWDjM6brhlQ)


![Example segmentation](example_segmentation.jpg?raw=true "Example segmentation")

# Packages
For instructions please refer to the README on each folder:

* [train](train) contains tools for training the network for semantic segmentation.
* [eval](eval) contains tools for evaluating/visualizing the network's output for panoramic annular semantic segmentation.
* [trained_models](trained_models) Contains the trained ERF-PSPNet and ERF-APSPNet models. 

# CODE Requirements:
* Dataset (I suggest using [**Mapillary Vistas**](https://www.mapillary.com/dataset/vistas) or Cityscpaes as training datasets. For evaluation, semantic segmentation on PASS, VISTAS or Cityscapes datasets can be tested using this code.) The Dataset shoudld be structured as the dataset folder indicates.
* [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I suggest installing it with [Anaconda](https://www.anaconda.com/download/#linux).
* [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code tested for CUDA 8.0, CUDA 9.0 and CUDA 10.0). I am using PyTorch 0.4.1.
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)

In Anaconda you can install with:
```
conda install numpy matplotlib torchvision Pillow
conda install -c conda-forge visdom
```

If you use Pip (make sure to have it configured for Python3.6) you can install with: 

```
pip install numpy matplotlib torchvision Pillow visdom
```

# Publications
If you use our code or dataset, please consider referencing any of the following papers:

**PASS: Panoramic Annular Semantic Segmentation.**
K. Yang, X. Hu, L.M. Bergasa, E. Romera, K. Wang.
IEEE Transactions on Intelligent Transportation Systems (**T-ITS**), September 2019.
[[**PDF**](http://www.robesafe.uah.es/personal/bergasa/papers/IEEE_T_ITS_Kailun_2019_FINAL%20VERSION.pdf)]

```
@article{yang2020pass,
title={PASS: Panoramic Annular Semantic Segmentation},
author={Yang, Kailun and Hu, Xinxin and Bergasa, Luis M and Romera, Eduardo and Wang, Kaiwei},
journal={IEEE Transactions on Intelligent Transportation Systems},
volume={21},
number={10},
pages={4171--4185},
year={2020},
publisher={IEEE}
}
```

**Can we PASS beyond the Field of View? Panoramic Annular Semantic Segmentation for Real-World Surrounding Perception.**
K. Yang, X. Hu, L.M. Bergasa, E. Romera, X. Huang, D. Sun, K. Wang.
In IEEE Intelligent Vehicles Symposium (**IV**), Paris, France, June 2019, pp. 446-453.
[[**PDF**](http://www.robesafe.uah.es/personal/bergasa/papers/IV2019_Kailun.pdf)]
[[**VIDEO**](http://www.robesafe.uah.es/personal/bergasa/papers/IV2019_1384x3432.mp4)]

```
@inproceedings{yang2019can,
title={Can we pass beyond the field of view? panoramic annular semantic segmentation for real-world surrounding perception},
author={Yang, Kailun and Hu, Xinxin and Bergasa, Luis M and Romera, Eduardo and Huang, Xiao and Sun, Dongming and Wang, Kaiwei},
booktitle={2019 IEEE Intelligent Vehicles Symposium (IV)},
pages={446--453},
year={2019},
organization={IEEE}
}
```
