import os

from benchmarks.multi_label_dataset.voc import MultiLabelVOC,MultiLabelVOC2,VOCForBatchLearning
from benchmarks.multi_label_dataset.nus_wide import NUS_WIDE,Imbalance_NUS_WIDE,NUSForBatchLearning
from benchmarks.multi_label_dataset.coco.mycoco import (MultiLabelCOCO, Imbalance_MultiLabelCOCO,
                                                        COCOForBatchLearning)
from torchvision.transforms import transforms
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_multi_dataset_generic_benchmark
import random
from pycocotools.coco import COCO
from tools.get_path import read_yaml

usr_root = os.path.expanduser("~")
config = read_yaml("paths.yaml")

dataset_root = usr_root + config["coco_dir"]

traindata_root = dataset_root + 'train/datas/'
train_annFile = dataset_root + 'annotations/instances_train2017.json'

valdata_root = dataset_root + 'validation/datas/'
val_annFile = dataset_root + 'annotations/instances_val2017.json'
global coco_train
global coco_test

sample_resolution = 224

def cl_ml_benchmark_from_different_datasets():
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.Resize([sample_resolution, sample_resolution]),
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([sample_resolution, sample_resolution]),
    ])

    voc_train_dataset = MultiLabelVOC(
        root= usr_root + config["voc_dir"],
        year="2012",
        image_set="train",
        transform=train_trans
    )
    voc_test_dataset = MultiLabelVOC(
        root= usr_root + config["voc_dir"],
        year="2012",
        image_set="val",
        transform=test_trans
    )

    nus_train_dataset = NUS_WIDE(transforms=train_trans, train=True)
    nus_test_dataset = NUS_WIDE(transforms=test_trans, train=False)

    coco_train_dataset = MultiLabelCOCO(train=True, transform=train_trans)
    coco_test_dataset = MultiLabelCOCO(train=False, transform=test_trans)

    benchmark = create_multi_dataset_generic_benchmark(
        train_datasets=(voc_train_dataset, nus_train_dataset, coco_train_dataset),
        test_datasets=(voc_test_dataset, nus_test_dataset, coco_test_dataset),
        task_labels=[0,1,2]
    )

    return benchmark


def multi_label_batch_learning_benchmark(dataset_name,seed):
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomVerticalFlip(0.5),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation(30),
        transforms.Resize([300, 300],antialias=True),
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([300, 300],antialias=True),
    ])
    if dataset_name == "voc":
        train_dataset = VOCForBatchLearning(
            root= usr_root + config["voc_dir"],
            year="2012",
            image_set="train",
            transform=train_trans
        )
        test_dataset = VOCForBatchLearning(
            root= usr_root + config["voc_dir"],
            year="2012",
            image_set="val",
            transform=test_trans
        )
    elif dataset_name == "nus":
        train_dataset = NUSForBatchLearning(transforms=train_trans, train=True)
        test_dataset = NUSForBatchLearning(transforms=test_trans, train=False)
    elif dataset_name == "coco":
        train_dataset = COCOForBatchLearning(train=True, transform=train_trans)
        test_dataset = COCOForBatchLearning(train=False, transform=test_trans)
    else:
        raise NotImplementedError


    benchmark = create_multi_dataset_generic_benchmark(
        train_datasets=(train_dataset,),
        test_datasets=(test_dataset,),
        task_labels=[0,]
    )

    return benchmark


def cl_ml_benckmark_from_single_dataset(dataset_name,task_num,use_aug):
    if use_aug=="yes":
        train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([300, 300]),
            transforms.RandomRotation(degrees=60,expand=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
    elif use_aug == "no":
        train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([300, 300]),
        ])
    else:
        raise NotImplementedError

    if dataset_name=="voc":
        classes_num = 20

    elif dataset_name=="coco":
        classes_num = 80

    elif dataset_name == "nus-wide":
        classes_num = 81

    else:
        raise NotImplementedError

    def get_dataset(dataset_name, task_classes, task_id):
        if dataset_name == "voc":
            train_dataset = MultiLabelVOC2(
                root= usr_root + config["voc_dir"],
                year="2012",
                image_set="trainval",
                transform=train_trans,
                task_classes=task_classes,
                task_id=task_id,
                train=True
            )
            test_dataset = MultiLabelVOC2(
                root=usr_root + config["voc_dir"],
                year="2012",
                image_set="trainval",
                transform=train_trans,
                task_classes=task_classes,
                task_id=task_id,
                train=False
            )
        elif dataset_name == "coco":
            train_dataset = MultiLabelCOCO(train=True, transform=train_trans,task_classes=task_classes,task_id=task_id)
            test_dataset = MultiLabelCOCO(train=False, transform=train_trans,task_classes=task_classes,task_id=task_id)
        elif dataset_name == "nus-wide":
            train_dataset = NUS_WIDE(transforms=train_trans, train=True,task_classes=task_classes,task_id=task_id)
            test_dataset = NUS_WIDE(transforms=train_trans, train=False,task_classes=task_classes,task_id=task_id)
        else:
            raise NotImplementedError
        return train_dataset,test_dataset

    # 制作每个任务所包含的类别数
    # 需要进一步考虑类别不平衡的情况
    task_classes = []
    class_ids = list(range(classes_num))
    random.shuffle(class_ids)
    step = int(len(class_ids) / task_num)
    for i in range(0, len(class_ids), step):
        task_classes.append(class_ids[i: i + step])

    train_datasets = {}
    test_datasets = {}
    for task_id,task_class in enumerate(task_classes):
        print("——"*60)
        print(f"Creating Task {task_id}")
        train_dataset,test_dataset = get_dataset(dataset_name,task_classes,task_id)
        train_datasets[task_id] = train_dataset
        test_datasets[task_id] = test_dataset


    benchmark = create_multi_dataset_generic_benchmark(
            train_datasets=(train_datasets),
            test_datasets=(test_datasets),
            task_labels=list(range(task_num)),
    )
    return benchmark,classes_num


def cl_ml_imbalance_benchmark_from_single_dataset(dataset_name,task_num,use_aug):
    if use_aug=="yes":
        train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([sample_resolution, sample_resolution]),
            transforms.RandomRotation(degrees=60,expand=False),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
    elif use_aug == "no":
        train_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([sample_resolution, sample_resolution],antialias=True),
        ])
    else:
        raise NotImplementedError

    if dataset_name=="voc":
        classes_num = 20

    elif dataset_name=="coco":
        classes_num = 80
        coco_train = COCO(train_annFile)
        coco_test = COCO(val_annFile)


    elif dataset_name == "nus":
        classes_num = 81

    else:
        raise NotImplementedError

    def get_dataset(dataset_name, task_classes, task_id):
        if dataset_name == "voc":
            train_dataset = MultiLabelVOC2(
                root=usr_root + config["voc_dir"],
                year="2012",
                image_set="trainval",
                transform=train_trans,
                task_classes=task_classes,
                task_id=task_id,
                train=True
            )
            test_dataset = MultiLabelVOC2(
                root=usr_root +config["voc_dir"],
                year="2012",
                image_set="trainval",
                transform=train_trans,
                task_classes=task_classes,
                task_id=task_id,
                train=False
            )
        elif dataset_name == "coco":
            train_dataset = Imbalance_MultiLabelCOCO(train=True,
                                                     transform=train_trans,
                                                     task_classes=task_classes,
                                                     task_id=task_id,
                                                     coco=coco_train)
            test_dataset = Imbalance_MultiLabelCOCO(train=False,
                                                    transform=train_trans,
                                                    task_classes=task_classes,
                                                    task_id=task_id,
                                                    coco=coco_test)
        elif dataset_name == "nus":
            train_dataset = Imbalance_NUS_WIDE(transforms=train_trans, train=True,task_classes=task_classes,task_id=task_id)
            test_dataset = Imbalance_NUS_WIDE(transforms=train_trans, train=False,task_classes=task_classes,task_id=task_id)
        else:
            raise NotImplementedError
        return train_dataset,test_dataset

    # 制作每个任务所包含的类别数
    # 需要进一步考虑类别不平衡的情况
    task_classes = []
    class_ids = list(range(classes_num))
    random.shuffle(class_ids)
    step = int(len(class_ids) / task_num)
    for i in range(0, len(class_ids), step):
        task_classes.append(class_ids[i: i + step])

    train_datasets = {}
    test_datasets = {}
    for task_id,task_class in enumerate(task_classes):
        print("——"*60)
        print(f"Creating Task {task_id}")
        train_dataset,test_dataset = get_dataset(dataset_name,task_classes,task_id)
        train_datasets[task_id] = train_dataset
        test_datasets[task_id] = test_dataset

        # calculate the imbalance level
        # targets = torch.stack(torch.unsqueeze(train_dataset.targets,1))
        # for c in targets.shape[1]:
        #     print("sa")
    benchmark = create_multi_dataset_generic_benchmark(
            train_datasets=(train_datasets),
            test_datasets=(test_datasets),
            task_labels=list(range(task_num)),
    )
    return benchmark,classes_num


if __name__ == '__main__':
    cl_ml_imbalance_benchmark_from_single_dataset("voc",task_num=5,use_aug="no")
    print("a")