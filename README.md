#  Non-learning SAMConvex registration method

This is the official Pytorch implementation of "SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid" (MICCAI 2023). If you have any questions, please contact us at alisonbrielee@gmail.com.


## Requirements
- Anaconda3 with python=3.7
- Pytorch=1.9.0
- [SAM: Self-supervised Learning of Pixel-wise Anatomical Embeddings in Radiological Images](https://ieeexplore.ieee.org/document/9760421/) [[repo]](https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2)


## Installation
First install SAM repo
```
pip install -U openmim
mim install mmcv-full==1.4.7
cd SAM
python -m pip install -e .
```
Then install this repo 
```
cd samconvex
pip install -e .
```


## Publication
If you find this repository useful, please cite:

- **SAMConvex: Fast Discrete Optimization for CT Registration using Self-supervised Anatomical Embedding and Correlation Pyramid**  
[Zi Li*](https://alison-brie.github.io/), Lin Tian*, Tony C. W. Mok, Xiaoyu Bai, Puyang Wang, Jia Ge, Jingren Zhou, Le Lu, Xianghua Ye, Ke Yan, Dakai Jin.
MICCAI 2023. [eprint arXiv:2307.09727](https://arxiv.org/abs/2307.09727 "eprint arXiv:2307.09727")
