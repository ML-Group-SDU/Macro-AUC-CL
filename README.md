**Pytorch implementation of "Towards Macro-AUC oriented Imbalanced Multi-Label Continual Learning"**

Our implementation is based on [Avalanche](https://github.com/ContinualAI/avalanche).

# Environment setup
Please create a python environment like `conda create -n mlcl python=3.10`.

We recommand using python=3.10, torch=2.1.0+cu121 and torchvision=0.16.0+cu121.
For other dependicies, please install them using `pip install -r requirements.txt`

# Datasets
Before running codes, please download the following multi-label datasets to your device and change the dataset path in `configs/paths.yaml`.

[PASCAL-VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

[MS-COCO 2017](https://cocodataset.org/)

[NUS-WIDE](https://lms.comp.nus.edu.sg/research/NUS-WIDE.html)

# Training instructions
Run the files in the directory '''run_commands''', for example: `python run_commands/rm_replay_based/voc/rm.py`.
The runing fils in this directory serves like bash files, they call the training python files in directory `train/`.
The results will be saved in directory `logs/`.