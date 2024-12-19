import json
import os.path
import random

from PIL import Image
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
import torch
from tools.get_path import get_project_path_and_name

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from typing import Any, Callable, Dict, Optional, Tuple, List


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class MultiLabelVOC(VOCDetection):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)


    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        target = self.get_onehot(target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_onehot(self, target):
        one_hot = torch.zeros(len(VOC_CLASSES))
        objects = target["annotation"]["object"]

        for item in objects:
            class_name = item["name"]
            class_id = VOC_CLASSES.index(class_name)
            one_hot[class_id] = 1

        return one_hot

    def get_num_for_class(self):
        t = torch.stack(self.targets,0)
        res = {}
        cur_classes_tensor = torch.tensor(self.cur_classes)
        for c in cur_classes_tensor:
            t_ = t[:,c]
            res[c] = [torch.sum(t_,0),t_.shape[0]-torch.sum(t_,0)]
        return res


class VOCForBatchLearning(VOCDetection):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.targets = [self.get_onehot(self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())) for index in range(self.__len__())]
        self.c_nums = self.get_num_for_class()
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        # target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        # target = self.get_onehot(target)
        target = self.targets[index]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_onehot(self, target):
        one_hot = torch.zeros(len(VOC_CLASSES))
        objects = target["annotation"]["object"]

        for item in objects:
            class_name = item["name"]
            class_id = VOC_CLASSES.index(class_name)
            one_hot[class_id] = 1

        return one_hot

    def get_num_for_class(self):
        t = torch.stack(self.targets, 0)
        res = {}
        cur_classes_tensor = torch.tensor(range(t[0].shape[0]))
        for c in cur_classes_tensor:
            t_ = t[:, c]
            res[c] = [torch.sum(t_, 0), t_.shape[0] - torch.sum(t_, 0)]
        return res

class MultiTaskVOC(VOCDetection):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            task_classes = None,
            task_id = None
    ):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.targets = [self.get_onehot(self.parse_voc_xml(ET_parse(i).getroot())) for i in self.annotations]

        # 找出所有带有目标类的data以及对应的target
        self.task_classes = task_classes
        self.task_id = task_id
        self.select_datas_for_classes()



    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        # target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        # target = self.get_onehot(target)
        target = self.targets[index]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_onehot(self, target):
        one_hot = torch.zeros(len(VOC_CLASSES))
        objects = target["annotation"]["object"]

        for item in objects:
            class_name = item["name"]
            class_id = VOC_CLASSES.index(class_name)
            one_hot[class_id] = 1

        return one_hot


    def cal_num_examples(self,task_num=5):
        target_array = torch.stack(self.targets)
        example_nums = torch.sum(target_array,0)

        task_classes = []
        task_dict = {}
        class_ids = list(range(20))
        random.shuffle(class_ids)
        step = int(len(class_ids) / task_num)
        for i in range(0, len(class_ids), step):
            task_classes.append(class_ids[i: i + step])

        for task_id,task_class in enumerate(task_classes):
            print(task_class)
            task_sum = None
            for class_id in task_class:
                if task_sum is None:
                    task_sum = target_array[:,class_id]
                else:
                    task_sum += target_array[:,class_id]
            a = torch.where(task_sum > 0,1,task_sum)
            task_dict[task_id] = a

        for k in task_dict.keys():
            if k < len(task_dict)-1:
                res = task_dict[k].bool() & task_dict[k+1].bool()

                print(sum(task_dict[k]))
                print(sum(task_dict[k+1]))
                print(sum(res))
                print("--"*30)

    def select_datas_for_classes(self):
        if self.task_classes is not None and self.task_id is not None:
            past_tasks = list(range(self.task_id)) if self.task_id > 0 else None
            self.seen_classes = []
            if past_tasks:
                self.seen_classes = [i for item in [self.task_classes[i] for i in past_tasks] for i in item]
            self.unseen_classes = list(set([i for item in self.task_classes for i in item]) - set(self.seen_classes) - set(
                self.task_classes[self.task_id]))

            target_array = torch.stack(self.targets)
            classes_array = target_array[:,self.task_classes[self.task_id]]
            res = torch.sum(classes_array,1)
            indexs = torch.where(res != 0)
            self.targets = [self.targets[i] for i in indexs[0]]

            # 把没见过的类和过去类的信息去掉，只保留当前类信息
            target_array = torch.stack(self.targets)
            for id in self.unseen_classes+self.seen_classes:
                target_array[:,id] = torch.zeros_like(target_array[:,id])

            self.class_nums = torch.sum(target_array, 0)
            self.targets = [target_array[i,:] for i in range(target_array.shape[0])]

            self.images = [self.images[i] for i in indexs[0]]


class MultiLabelVOC2(VOCDetection):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            task_classes = None,
            task_id = None,
            train=True
    ):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)


        self.imb_root = get_project_path_and_name()[0] + "saves/voc/imbalance/"
        if not os.path.exists(self.imb_root):
            os.makedirs(self.imb_root)
        if os.path.exists(self.imb_root + "trainval_targets.pt"):
            self.targets = torch.load(self.imb_root+"trainval_targets.pt")
            self.targets = [torch.squeeze(e) for e in torch.split(self.targets,1,0)]
        else:
            self.targets = [self.get_onehot(self.parse_voc_xml(ET_parse(i).getroot())) for i in self.annotations]
            torch.save(torch.stack(self.targets), self.imb_root+"trainval_targets.pt")

        self.pt_name = str(task_classes[task_id])+".pt"

        # 找出所有带有目标类的data以及对应的target
        self.task_classes = task_classes
        self.task_id = task_id
        self.select_datas_for_classes()

        dataset_len = len(self.images)
        dataset_idxs = list(range(dataset_len))
        random.shuffle(dataset_idxs)
        if train==True:
            idxs = dataset_idxs[:int(dataset_len*0.5)]
        else:
            idxs = dataset_idxs[int(dataset_len*0.5):]

        self.targets = [self.targets[i] for i in idxs]
        self.images = [self.images[i] for i in idxs]
        print(f"len_targets:{len(self.targets)},dataset:{str(train)}",)
        self.c_nums = self.get_num_for_class()
        # print("c_nums:",self.c_nums)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        # target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        # target = self.get_onehot(target)
        target = self.targets[index]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_onehot(self, target):
        one_hot = torch.zeros(len(VOC_CLASSES))
        objects = target["annotation"]["object"]

        for item in objects:
            class_name = item["name"]
            class_id = VOC_CLASSES.index(class_name)
            one_hot[class_id] = 1

        return one_hot

    def cal_num_examples(self,task_num=5):
        target_array = torch.stack(self.targets)
        example_nums = torch.sum(target_array,0)

        task_classes = []
        task_dict = {}
        class_ids = list(range(20))
        random.shuffle(class_ids)
        step = int(len(class_ids) / task_num)
        for i in range(0, len(class_ids), step):
            task_classes.append(class_ids[i: i + step])

        for task_id,task_class in enumerate(task_classes):
            print(task_class)
            task_sum = None
            for class_id in task_class:
                if task_sum is None:
                    task_sum = target_array[:,class_id]
                else:
                    task_sum += target_array[:,class_id]
            a = torch.where(task_sum > 0,1,task_sum)
            task_dict[task_id] = a

        for k in task_dict.keys():
            if k < len(task_dict)-1:
                res = task_dict[k].bool() & task_dict[k+1].bool()

                print(sum(task_dict[k]))
                print(sum(task_dict[k+1]))
                print(sum(res))
                print("--"*30)

    def select_datas_for_classes(self):
        if self.task_classes is not None and self.task_id is not None:
            past_tasks = list(range(self.task_id)) if self.task_id > 0 else None
            self.seen_classes = []
            if past_tasks:
                self.seen_classes = [i for item in [self.task_classes[i] for i in past_tasks] for i in item]
            self.unseen_classes = list(set([i for item in self.task_classes for i in item]) - set(self.seen_classes) - set(
                self.task_classes[self.task_id]))
            self.cur_classes = self.task_classes[self.task_id]

            target_array = torch.stack(self.targets)
            classes_array = target_array[:,self.cur_classes]
            res = torch.sum(classes_array,1)

            if os.path.exists(self.imb_root+self.pt_name):
                self.task_specfic_indexs = [torch.load(self.imb_root+self.pt_name)]
            else:
                self.task_specfic_indexs = torch.where(res != 0) # 当前任务涉及到的样本索引

            self.targets = [self.targets[i] for i in self.task_specfic_indexs[0]]

            # 把没见过的类和过去类的信息去掉，只保留当前类信息
            target_array = torch.stack(self.targets)
            for id in self.unseen_classes+self.seen_classes:
                target_array[:,id] = torch.zeros_like(target_array[:,id])

            self.class_nums = torch.sum(target_array, 0)

            self.targets = [target_array[i,:] for i in range(target_array.shape[0])]
            self.images = [self.images[i] for i in self.task_specfic_indexs[0]]
            if not os.path.exists(self.imb_root+self.pt_name):
                self.reset_classes_to_imbalance(target_array)

    def reset_classes_to_imbalance(self,target_array):
        def cal_nums(target_array):
            res = {}
            cur_classes_tensor = torch.tensor(self.cur_classes)
            for c in cur_classes_tensor:
                t_ = target_array[:, c]
                res[c.item()] = torch.sum(t_, 0)
            return res

        res = cal_nums(target_array)
        res_sort_list = sorted(res.items(), key=lambda s: s[1])
        new_res = {}
        for i in res_sort_list:
            r = i[1]/res_sort_list[-1][1]
            if r.item() < 1:
                r = r * (random.randrange(5,100)/100)
            new_res[i[0]] = torch.tensor(int(r * res_sort_list[-1][-1]))

        def dis(res1:dict,res2:dict):
            d = []
            for k in res1.keys():
                d.append(torch.abs(torch.sub(res1[k],res2[k])))
            return torch.mean(torch.tensor(d))

        # 贪心算法选择样本删去
        d = dis(res,new_res)
        last_d = None
        idx_list = list(range(target_array.shape[0]))
        count = 0
        while count < 10:
            tmp_idx = None
            sub_list = random.sample(idx_list,100)
            for idx in sub_list:
                idx_list.remove(idx)
                tmp_res = cal_nums(target_array[torch.tensor(idx_list),:])
                tmp_d = dis(tmp_res,new_res)
                if tmp_d < d:
                    d = tmp_d
                    tmp_idx = idx
                idx_list.append(idx)
            if tmp_idx:
                idx_list.remove(tmp_idx)
            print(d)
            if d == last_d:
                count = count+1
            else:
                count = 0
            last_d = d

        self.targets = [self.targets[i] for i in idx_list]
        self.images = [self.images[i] for i in idx_list]

        nums = [i.item() for i in tmp_res.values()]
        with open(self.imb_root+str(self.cur_classes)+".txt", 'w+', encoding='utf-8') as f:
            f.write(str(nums))

        sss = self.task_specfic_indexs[0][torch.tensor(idx_list)]
        torch.save(sss,self.imb_root+self.pt_name)

    def get_num_for_class(self):
        t = torch.stack(self.targets,0)
        res = {}
        cur_classes_tensor = torch.tensor(self.cur_classes)
        for c in cur_classes_tensor:
            t_ = t[:,c]
            res[c] = [torch.sum(t_,0),t_.shape[0]-torch.sum(t_,0)]
        return res


class MultiLabelVOC2ori(VOCDetection):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            task_classes = None,
            task_id = None,
    ):
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)


        self.imb_root = get_project_path_and_name()[0] + f"saves/voc/imbalance/{image_set}/"
        if not os.path.exists(self.imb_root):
            os.makedirs(self.imb_root)
        if os.path.exists(self.imb_root+f"{image_set}_targets.pt"):
            self.targets = torch.load(self.imb_root+f"{image_set}_targets.pt")
            self.targets = [torch.squeeze(e) for e in torch.split(self.targets,1,0)]
        else:
            self.targets = [self.get_onehot(self.parse_voc_xml(ET_parse(i).getroot())) for i in self.annotations]
            torch.save(torch.stack(self.targets), self.imb_root+f"{image_set}_targets.pt")

        self.pt_name = str(task_classes[task_id])+".pt"

        # 找出所有带有目标类的data以及对应的target
        self.task_classes = task_classes
        self.task_id = task_id
        self.select_datas_for_classes()

        # dataset_len = len(self.images)
        # dataset_idxs = list(range(dataset_len))
        # random.shuffle(dataset_idxs)
        # if train==True:
        #     idxs = dataset_idxs[:int(dataset_len*0.5)]
        # else:
        #     idxs = dataset_idxs[int(dataset_len*0.5):]
        #
        # self.targets = [self.targets[i] for i in idxs]
        # self.images = [self.images[i] for i in idxs]
        self.c_nums = self.get_num_for_class()

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        # target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        # target = self.get_onehot(target)
        target = self.targets[index]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_onehot(self, target):
        one_hot = torch.zeros(len(VOC_CLASSES))
        objects = target["annotation"]["object"]

        for item in objects:
            class_name = item["name"]
            class_id = VOC_CLASSES.index(class_name)
            one_hot[class_id] = 1

        return one_hot

    def cal_num_examples(self,task_num=5):
        target_array = torch.stack(self.targets)
        example_nums = torch.sum(target_array,0)

        task_classes = []
        task_dict = {}
        class_ids = list(range(20))
        random.shuffle(class_ids)
        step = int(len(class_ids) / task_num)
        for i in range(0, len(class_ids), step):
            task_classes.append(class_ids[i: i + step])

        for task_id,task_class in enumerate(task_classes):
            print(task_class)
            task_sum = None
            for class_id in task_class:
                if task_sum is None:
                    task_sum = target_array[:,class_id]
                else:
                    task_sum += target_array[:,class_id]
            a = torch.where(task_sum > 0,1,task_sum)
            task_dict[task_id] = a

        for k in task_dict.keys():
            if k < len(task_dict)-1:
                res = task_dict[k].bool() & task_dict[k+1].bool()

                print(sum(task_dict[k]))
                print(sum(task_dict[k+1]))
                print(sum(res))
                print("--"*30)

    def select_datas_for_classes(self):
        if self.task_classes is not None and self.task_id is not None:
            past_tasks = list(range(self.task_id)) if self.task_id > 0 else None
            self.seen_classes = []
            if past_tasks:
                self.seen_classes = [i for item in [self.task_classes[i] for i in past_tasks] for i in item]
            self.unseen_classes = list(set([i for item in self.task_classes for i in item]) - set(self.seen_classes) - set(
                self.task_classes[self.task_id]))
            self.cur_classes = self.task_classes[self.task_id]

            target_array = torch.stack(self.targets)
            classes_array = target_array[:,self.cur_classes]
            res = torch.sum(classes_array,1)

            if os.path.exists(self.imb_root+self.pt_name):
                self.task_specfic_indexs = [torch.load(self.imb_root+self.pt_name)]
            else:
                self.task_specfic_indexs = torch.where(res != 0) # 当前任务涉及到的样本索引

            self.targets = [self.targets[i] for i in self.task_specfic_indexs[0]]

            # 把没见过的类和过去类的信息去掉，只保留当前类信息
            target_array = torch.stack(self.targets)
            for id in self.unseen_classes+self.seen_classes:
                target_array[:,id] = torch.zeros_like(target_array[:,id])

            self.class_nums = torch.sum(target_array, 0)

            self.targets = [target_array[i,:] for i in range(target_array.shape[0])]
            self.images = [self.images[i] for i in self.task_specfic_indexs[0]]
            if not os.path.exists(self.imb_root+self.pt_name):
                if self.image_set!="val":
                    self.reset_classes_to_imbalance(target_array)

    def reset_classes_to_imbalance(self,target_array):
        def cal_nums(target_array):
            res = {}
            cur_classes_tensor = torch.tensor(self.cur_classes)
            for c in cur_classes_tensor:
                t_ = target_array[:, c]
                res[c.item()] = torch.sum(t_, 0)
            return res

        res = cal_nums(target_array)
        res_sort_list = sorted(res.items(), key=lambda s: s[1])
        new_res = {}
        for i in res_sort_list:
            r = i[1]/res_sort_list[-1][1]
            if r.item() < 1:
                r = r * (random.randrange(5,100)/100)
            new_res[i[0]] = torch.tensor(int(r * res_sort_list[-1][-1]))

        def dis(res1:dict,res2:dict):
            d = []
            for k in res1.keys():
                d.append(torch.abs(torch.sub(res1[k],res2[k])))
            return torch.mean(torch.tensor(d))

        # 贪心算法选择样本删去
        d = dis(res,new_res)
        last_d = None
        idx_list = list(range(target_array.shape[0]))
        count = 0
        while count < 10:
            tmp_idx = None
            sub_list = random.sample(idx_list,100)
            for idx in sub_list:
                idx_list.remove(idx)
                tmp_res = cal_nums(target_array[torch.tensor(idx_list),:])
                tmp_d = dis(tmp_res,new_res)
                if tmp_d < d:
                    d = tmp_d
                    tmp_idx = idx
                idx_list.append(idx)
            if tmp_idx:
                idx_list.remove(tmp_idx)
            print(d)
            if d == last_d:
                count = count+1
            else:
                count = 0
            last_d = d

        self.targets = [self.targets[i] for i in idx_list]
        self.images = [self.images[i] for i in idx_list]

        nums = [i.item() for i in tmp_res.values()]
        with open(self.imb_root+str(self.cur_classes)+".txt", 'w+', encoding='utf-8') as f:
            f.write(str(nums))

        sss = self.task_specfic_indexs[0][torch.tensor(idx_list)]
        torch.save(sss,self.imb_root+self.pt_name)

    def get_num_for_class(self):
        t = torch.stack(self.targets,0)
        res = {}
        cur_classes_tensor = torch.tensor(self.cur_classes)
        for c in cur_classes_tensor:
            t_ = t[:,c]
            res[c] = [torch.sum(t_,0),t_.shape[0]-torch.sum(t_,0)]
        return res


if __name__ == '__main__':
    train_trans = transforms.ToTensor()

    dataset = MultiLabelVOC2(
        root= os.path.expanduser("~")+"/data/Datasets/VOC/",
        year="2012",
        image_set="trainval",
        transform=train_trans,
        task_classes=[[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15],[16,17,18,19]],
        task_id=0
    )
    #
    # dataset.cal_num_examples()
