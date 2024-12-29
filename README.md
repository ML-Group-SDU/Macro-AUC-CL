# Towards Macro-AUC oriented Imbalanced Multi-Label Continual Learning

**Pytorch implementation of AAAI 2025 paper.**

Yan Zhang, Guoqiang Wu, Bingzheng Wang, Teng Pang, Haoliang Sun, Yilong Yin

https://arxiv.org/abs/2412.18231

Abstract:
*In Continual Learning (CL), while existing work primarily focuses on the multi-class classification task, there has been limited research on Multi-Label Learning (MLL). In practice, MLL datasets are often class-imbalanced, making it inherently challenging, a problem that is even more acute in CL. 
Due to its sensitivity to imbalance, Macro-AUC is an appropriate and widely used measure in MLL. 
However, there is no research to optimize Macro-AUC in MLCL specifically.
To fill this gap, in this paper, we propose a new memory replay-based method to tackle the imbalance issue for Macro-AUC-oriented MLCL.
Specifically, inspired by recent theory work, we propose a new Reweighted Label-Distribution-Aware Margin (RLDAM) loss.
Furthermore, to be compatible with the RLDAM loss, a new memory-updating strategy named Weight Retain Updating (WRU) is proposed to maintain the numbers of positive and negative instances of the original dataset in memory. 
Theoretically, we provide superior generalization analyses of the RLDAM-based algorithm in terms of Macro-AUC, separately in batch MLL and MLCL settings. This is the first work to offer theoretical generalization analyses in MLCL to our knowledge.
Finally, a series of experimental results illustrate the effectiveness of our method over several baselines.*

# Requirements
- Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.

- We have done all training and testing and development using GeForce RTX 3090 and 4090 GPUs.

- 64-bit Python 3.10, PyTorch 2.1.0, Torchvision 0.16.0 and CUDA 12.1.
- See environmental.yml and requirements.txt for exact library dependencies. You can create your python environment using the following commands:
  - `conda create -n mlcl python=3.10`
  - `pip install -r requirements.txt`
  - or
  - `conda env create -f environmental.yml -n mlcl`
  - `conda activate mlcl`

# Datasets
Before running codes, please download the following multi-label datasets to your device and change the dataset path in `configs/paths.yaml`.

[PASCAL-VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

[MS-COCO 2017](https://cocodataset.org/)

[NUS-WIDE](https://lms.comp.nus.edu.sg/research/NUS-WIDE.html)

# Training instructions
Run the files in the directory '''run_commands''', for example: 

`python run_commands/rm_replay_based/voc/rm_voc.py`.

The runing fils in this directory serves like bash files, they call the training python files in directory `train/`.
The results will be saved in directory `logs/`.

# Acknowledgements
Our implementation is based on [Avalanche](https://github.com/ContinualAI/avalanche).

# Citation
```
@misc{zhang2024mlcl,
      title={Towards Macro-AUC oriented Imbalanced Multi-Label Continual Learning}, 
      author={Yan Zhang and Guoqiang Wu and Bingzheng Wang and Teng Pang and Haoliang Sun and Yilong Yin},
      year={2024},
      eprint={2412.18231},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.18231}, 
}
```

