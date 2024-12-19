import pickle
from os.path import expanduser
import os
import torch

from avalanche.benchmarks.datasets import CIFAR100
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
from torch.utils.data import Dataset
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class CIFAR_Dataset10(Dataset):
    def __init__(self,data_dir,train,transform):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

        if self.train:
            for i in range(5):
                with open(data_dir+"data_batch_"+str(i+1),'rb') as f:
                    entry = pickle.load(f,encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1,3,32,32)
        self.data = self.data.transpose((0,2,3,1))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = torch.zeros(10)
        label[self.targets[idx]] = 1

        if self.transform:
            image = icarl_cifar100_augment_data(self.data[idx])

        if self.train and idx > 0 and idx % 5==0:
            mixup_idx = random.randint(0,len(self.data)-1)
            mixup_label = torch.zeros(10)
            label[self.targets[mixup_idx]] = 1
            if self.transform:
                mixup_image = icarl_cifar100_augment_data(self.data[mixup_idx])

            alpha = 0.2
            lam = np.random.beta(alpha,alpha)
            image = lam*image + (1-lam)*mixup_image
            label = lam*label + (1-lam)*mixup_label

        return image,label

class CIFAR_Dataset100(Dataset):
    def __init__(self,data_dir,train,transform):
        self.train = train  # training set or test set
        self.transform = transform
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        file_name = "cifar-100-python/train"
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]


        one_hot_label = torch.zeros(100)
        one_hot_label[target] = 1

        # if self.train and index > 0:
        #     mixup_idx = random.randint(0,len(self.data)-1)
        #     mixup_onehot = torch.zeros(100)
        #     mixup_onehot[self.targets[mixup_idx]] = 1
        #     mixup_image = self.data[mixup_idx]
        #
        #     lam = np.random.beta(3.0,0.3)
        #
        #     img = lam*img + (1-lam)*mixup_image
        #     one_hot_label = lam * one_hot_label + (1-lam) * mixup_onehot

        img = Image.fromarray(np.uint8(img))

        if self.transform:
            img = self.transform(img)
        return img, target, one_hot_label


def get_dataset_per_pixel_mean(dataset):
    result = None
    patterns_count = 0

    for img_pattern, _,_ in dataset:
        if result is None:
            result = torch.zeros_like(img_pattern, dtype=torch.float)

        result += img_pattern
        patterns_count += 1

    if result is None:
        result = torch.empty(0, dtype=torch.float)
    else:
        result = result / patterns_count

    return result


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    per_pixel_mean = get_dataset_per_pixel_mean(
        CIFAR_Dataset100(
            expanduser("~") + "/.avalanche/data/cifar100/",
            train=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    )
    transforms_group = dict(
        eval=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                ]
            ),
            None,
        ),
        train=(
            transforms.Compose(
                [
                    transforms.ToTensor(),
                    lambda img_pattern: img_pattern - per_pixel_mean,
                    icarl_cifar100_augment_data,
                ]
            ),
            None,
        ),
    )

    train_set = CIFAR_Dataset100(
        expanduser("~") + "/.avalanche/data/cifar100/",
        train=True,
        transform=False
    )
    # train_set = CIFAR100(
    #     expanduser("~") + "/.avalanche/data/cifar100/",
    #     train=True,
    #     download=True,
    # )
    test_set = CIFAR100(
        expanduser("~") + "/.avalanche/data/cifar100/",
        train=False,
        download=True,
    )

    train_set = AvalancheDataset(
        train_set,
        transform_groups=transforms_group,
        initial_transform_group="train",
    )


    test_set = AvalancheDataset(
        test_set,
        transform_groups=transforms_group,
        initial_transform_group="eval",
    )


    scenario = nc_benchmark(
        train_dataset=train_set,
        test_dataset=test_set,
        n_experiences=config.nb_exp,
        task_labels=False,
        seed=config.seed,
        shuffle=False,
        fixed_class_order=config.fixed_class_order,
    )
    os.makedirs(config.log_dir, exist_ok=True)
    evaluator = EvaluationPlugin(
        EpochAccuracy(),
        ExperienceAccuracy(),
        StreamAccuracy(),
        loggers=[InteractiveLogger(),
                 TextLogger(open(config.log_dir+config.log_file_name, 'a+'))],
    )

    model: IcarlNet = make_icarl_net(num_classes=100)
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
    config.nb_exp = 10
    config.memory_size = 2000
    config.epochs = 80
    config.lr_base = 2.0
    config.lr_milestones = [49, 63]
    config.lr_factor = 5.0
    config.wght_decay = 0.00001
    config.fixed_class_order = [
        87,
        0,
        52,
        58,
        44,
        91,
        68,
        97,
        51,
        15,
        94,
        92,
        10,
        72,
        49,
        78,
        61,
        14,
        8,
        86,
        84,
        96,
        18,
        24,
        32,
        45,
        88,
        11,
        4,
        67,
        69,
        66,
        77,
        47,
        79,
        93,
        29,
        50,
        57,
        83,
        17,
        81,
        41,
        12,
        37,
        59,
        25,
        20,
        80,
        73,
        1,
        28,
        6,
        46,
        62,
        82,
        53,
        9,
        31,
        75,
        38,
        63,
        33,
        74,
        27,
        22,
        36,
        3,
        16,
        21,
        60,
        19,
        70,
        90,
        89,
        43,
        5,
        42,
        65,
        76,
        40,
        30,
        23,
        85,
        2,
        95,
        56,
        48,
        71,
        64,
        98,
        13,
        99,
        7,
        34,
        55,
        54,
        26,
        35,
        39,
    ]
    config.seed = 2222
    config.log_dir = "./replay_logs/icarl/"
    config.log_file_name = "icarl10-mixup_buffer3.txt"

    run_experiment(config)
