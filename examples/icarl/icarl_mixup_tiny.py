"""
用的自带的Split-Tiny，而不是从头重新构建scenario
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from os.path import expanduser
import torch
from avalanche.benchmarks.datasets import tiny_imagenet
from avalanche.benchmarks import SplitTinyImageNet
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import IcarlNet, make_icarl_net, initialize_icarl_net
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch.optim import SGD
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
            :, crop[0] : (crop[0] + 64), crop[1] : (crop[1] + 64)
        ]
    else:
        random_cropped[:, :, :] = padded[
            :, crop[0] : (crop[0] + 64), crop[1] : (crop[1] + 64)
        ][:, :, ::-1]
    t = torch.tensor(random_cropped)
    return t

def run_experiment(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else None)
    fix_seeds(config.seed)

    scenario = SplitTinyImageNet(n_experiences=config.nb_exp,
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

    model: IcarlNet = make_icarl_net(num_classes=200)
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
        mixup=True,
    )

    for i, exp in enumerate(scenario.train_stream):
        eval_exps = [e for e in scenario.test_stream][: i + 1]
        strategy.train(exp, num_workers=4)
        strategy.eval(eval_exps, num_workers=4)


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
    config.nb_exp = 20
    config.memory_size = 2000
    config.epochs = 70
    config.lr_base = 2.0
    config.lr_milestones = [49, 63]
    config.lr_factor = 5.0
    config.wght_decay = 0.00001
    config.fixed_class_order = [195, 144, 36, 11, 170, 153, 44, 72, 15, 168, 126, 151, 70, 130,
                         137, 65, 22, 92, 1, 42, 9, 8, 19, 115, 77, 123, 118, 182,
                         108, 160, 178, 93, 80, 18, 48, 2, 99, 122, 24, 152, 63, 179,
                         166, 75, 69, 154, 159, 119, 172, 14, 162, 31, 40, 135, 184, 94,
                         158, 25, 89, 68, 147, 155, 175, 98, 186, 60, 134, 67, 97, 197,
                         86, 56, 109, 34, 106, 96, 164, 28, 58, 143, 128, 90, 145, 102,
                         176, 38, 189, 190, 121, 57, 51, 55, 163, 138, 169, 188, 177, 104,
                         148, 61, 136, 114, 174, 111, 79, 7, 13, 87, 192, 101, 54, 180,
                         113, 161, 194, 193, 83, 142, 131, 129, 64, 171, 105, 146, 73, 112,
                         29, 37, 12, 82, 78, 150, 199, 81, 95, 45, 185, 141, 157, 39,
                         10, 125, 43, 107, 198, 76, 26, 120, 117, 173, 187, 140, 110, 181,
                         4, 33, 183, 62, 49, 17, 91, 50, 127, 6, 196, 132, 47, 53,
                         30, 156, 27, 5, 3, 165, 71, 116, 133, 88, 167, 0, 16, 23,
                         191, 35, 59, 32, 85, 46, 74, 21, 103, 52, 100, 149, 20, 124,
                         139, 66, 41, 84]
    config.seed = 2222
    config.log_dir = "./replay_logs/icarl/"
    config.log_file_name = "icarl_tinyimagenet_mixup.txt"

    run_experiment(config)
