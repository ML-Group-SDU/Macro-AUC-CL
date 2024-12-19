"""
用的自带的Split-Cifar10，而不是从头重新构建scenario
"""

import os
from os.path import expanduser
import torch
from avalanche.benchmarks import SplitMNIST
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD,Adam
from torchvision import transforms
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    ExperienceAccuracy,
    StreamAccuracy,
    EpochAccuracy,
)
from avalanche.logging.interactive_logging import InteractiveLogger,TextLogger
import random
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from avalanche.training.supervised.icarl import ICaRL
from examples.fix_seeds import fix_seeds

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def icarl_cifar100_augment_data(img):
    img = img.numpy()
    padded = np.pad(img, ((0, 0), (4, 4), (4, 4)), mode="constant")
    random_cropped = np.zeros(img.shape, dtype=np.float32)
    crop = np.random.randint(0, high=8 + 1, size=(2,))

    # Cropping and possible flipping
    if np.random.randint(2) > 0:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ]
    else:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 32), crop[1] : (crop[1] + 32)
        ][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t

def run_experiment(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else None)
    fix_seeds(config.seed)

    scenario = SplitMNIST(n_experiences=config.nb_exp,
                            seed=config.seed,
                            shuffle=False,
                            fixed_class_order=config.fixed_class_order,
                            )

    evaluator = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[InteractiveLogger(),
                 TextLogger(open(config.log_dir+config.log_file_name,'a+'))],
    )

    model: IcarlNet = make_icarl_net(num_classes=10)
    model.apply(initialize_icarl_net)

    optim = SGD(
        model.parameters(),
        lr=config.lr_base,
        weight_decay=config.wght_decay,
        momentum=0.9,
    )
    sched = LRSchedulerPlugin(
        MultiStepLR(optim, config.lr_milestones, gamma=1.0 / config.lr_factor)
    )

    strategy = ICaRL(
        model.feature_extractor,
        model.classifier,
        optim,
        config.memory_size,
        buffer_transform=transforms.Compose([icarl_cifar100_augment_data]),
        fixed_memory=True,
        train_mb_size=config.batch_size,
        train_epochs=config.epochs,
        eval_mb_size=config.batch_size,
        plugins=[sched],
        device=device,
        evaluator=evaluator,
    )

    for i, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: i + 1]
        strategy.train(exp, num_workers=2)
        strategy.eval(eval_exps, num_workers=2)


class Config(dict):
    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == "__main__":
    config = Config()

    config.batch_size = 128
    config.nb_exp = 5
    config.memory_size = 2000
    config.epochs = 35
    config.lr_base = 0.2
    config.lr_milestones = [19, 27]
    config.lr_factor = 5.0
    config.wght_decay = 0.00001
    config.fixed_class_order = [0,6,9,4,5,7,8,1,2,3]
    config.seed = 2222
    config.log_dir = "./replay_logs/icarl/"
    config.log_file_name = "icarl_smnist_general.txt"

    run_experiment(config)
