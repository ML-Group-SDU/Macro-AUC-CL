from typing import Tuple, Any, Optional, Sequence, Union

import torch
from PIL import Image
from torchvision.datasets.cifar import CIFAR10, CIFAR100

from avalanche.benchmarks import nc_benchmark, NCScenario
from avalanche.benchmarks.datasets import TinyImagenet, default_dataset_location
import numpy as np
import math
from tools.get_path import *
import json
from torchvision import transforms
from pathlib import Path

class BinaryCIFAR10(CIFAR10):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 two_class_id=None,
                 scalers=None
                 ):
        super().__init__(root, train, transform, target_transform, download)

        scalers = scalers.split(",")
        scalers = [float(scalers[i].strip()) for i in range(len(scalers))]
        if two_class_id is not None:
            class1, class2 = two_class_id[0], two_class_id[1]
        else:
            class1,class2 = 0,1
        dataset_len = len(self.data)
        data_dic = {}
        for class_id in [class1,class2]:
            data_dic[class_id] = []

        for i in range(dataset_len):
            if self.targets[i] == class1:
                data_dic[class1].append(self.data[i])
            elif self.targets[i] == class2:
                data_dic[class2].append(self.data[i])
            else:
                pass

        num_per_class = {}
        for cls_id, v in data_dic.items():
            scale_factor = scalers[cls_id]
            num = math.ceil(scale_factor * len(v))
            num_per_class[cls_id] = num

        print("num_per_class: ",num_per_class)

        for k, v in num_per_class.items():
            random_indexs = np.random.randint(low=0, high=len(data_dic[k]), size=v)
            data_dic[k] = np.stack(data_dic[k])
            data_dic[k] = data_dic[k][random_indexs]

        targets = []
        datas = []
        for k, v in data_dic.items():
            target_tmp = [k] * len(v)
            datas.extend(v)
            targets.extend(target_tmp)
        self.data = datas
        self.targets = targets


class BinaryCIFAR100(CIFAR100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False, ):
        super().__init__(root=root,train=train,transform=transform,target_transform=target_transform,download=download)

        dataset_len = len(self.data)
        data_dic = {}
        for class_id in set(self.targets):
            data_dic[class_id] = []
        for i in range(dataset_len):
            class_id = self.targets[i]
            data_dic[class_id].append(self.data[i])

        with open(get_project_path() + "/saves/cifar100.json", "r") as f2:
            scalers = json.load(f2)

        num_per_class = {}
        for cls_id, v in data_dic.items():
            scale_factor = scalers[str(cls_id)]
            num = math.ceil(scale_factor * len(v))
            num_per_class[cls_id] = num

        for k, v in num_per_class.items():
            random_indexs = np.random.randint(low=0, high=len(data_dic[k]), size=v)
            data_dic[k] = np.stack(data_dic[k])
            data_dic[k] = data_dic[k][random_indexs]

        targets = []
        datas = []
        for k, v in data_dic.items():
            target_tmp = [k] * len(v)
            datas.extend(v)
            targets.extend(target_tmp)
        self.data = datas
        self.targets = targets


class BinaryTinyImageNet(TinyImagenet):
    def load_data(self):
        data = [[], []]

        classes = list(range(200))

        with open(get_project_path()+"/saves/tinyimagenet.json", "r") as f2:
            scalers = json.load(f2)

        for class_id in classes:
            class_name = self.id2label[class_id]

            if self.train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            sample_nums = math.ceil(np.round(scalers[str(class_id)] * len(Y), 2))
            random_indexs = np.random.randint(low=0, high=len(Y), size=sample_nums)

            X = [X[id] for id in random_indexs]
            Y = [Y[id] for id in random_indexs]

            data[0] += X
            data[1] += Y

        return data

_default_cifar10_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

_default_cifar10_eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

def get_cifar10_dataset(dataset_root,two_class_id,train_scalers,test_scalers):
    if dataset_root is None:
        dataset_root = default_dataset_location("cifar10")

    train_set = BinaryCIFAR10(dataset_root, train=True, download=True,two_class_id=two_class_id,scalers=train_scalers)
    test_set = BinaryCIFAR10(dataset_root, train=False, download=True,two_class_id=two_class_id,scalers=test_scalers)

    return train_set, test_set

def BinarySplitCIFAR10(
        n_experiences: int,
        *,
        first_exp_with_half_classes: bool = False,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        class_ids_from_zero_in_each_exp: bool = False,
        train_transform: Optional[Any] = _default_cifar10_train_transform,
        eval_transform: Optional[Any] = _default_cifar10_eval_transform,
        dataset_root: Union[str, Path] = None,
        train_error: bool=False,
        two_class_id,
        train_scalers,
        test_scalers,
) -> NCScenario:
    """
    Creates a CL benchmark using the CIFAR10 dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences in the current benchmark.
        The value of this parameter should be a divisor of 10 if
        `first_task_with_half_classes` is False, a divisor of 5 otherwise.
    :param first_exp_with_half_classes: A boolean value that indicates if a
        first pretraining step containing half of the classes should be used.
        If it's True, the first experience will use half of the classes (5 for
        cifar10). If this parameter is False, no pretraining step will be
        used and the dataset is simply split into a the number of experiences
        defined by the parameter n_experiences. Defaults to False.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If not None, the ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to True.
    :param class_ids_from_zero_in_each_exp: If True, original class IDs
        will be mapped to range [0, n_classes_in_exp) for each experience.
        Defaults to False. Mutually exclusive with the
        ``class_ids_from_zero_from_first_exp`` parameter.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default eval transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'cifar10' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    cifar_train, cifar_test = get_cifar10_dataset(dataset_root,two_class_id,train_scalers,test_scalers)

    return nc_benchmark(
        train_dataset=cifar_train,
        test_dataset=cifar_test if not train_error else cifar_train,
        n_experiences=n_experiences,
        task_labels=return_task_id,
        seed=seed,
        fixed_class_order=fixed_class_order,
        shuffle=shuffle,
        per_exp_classes={0: 5} if first_exp_with_half_classes else None,
        class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )




if __name__ == '__main__':
    a = BinarySplitCIFAR10(n_experiences=1,
                           seed=2222)