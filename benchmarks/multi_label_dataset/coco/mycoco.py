# coding=utf-8
import os.path
from typing import Optional, Callable,List,Any,Tuple

import torch.utils.data
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
import shutil
from benchmarks.multi_label_dataset.coco.coco_classes import *
from tools.get_path import get_project_path_and_name
from avalanche.models.transformer import *
import torch.nn as nn
from PIL import Image
from tools.get_path import read_yaml

configs = read_yaml("paths.yaml")
usr_root = os.path.expanduser("~")
dataset_root = usr_root + configs["coco_dir"]

traindata_root = dataset_root + 'train/data/'
train_annFile = dataset_root + 'annotations/instances_train2017.json'

valdata_root = dataset_root + 'validation/data/'
val_annFile = dataset_root + 'annotations/instances_val2017.json'

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        coco = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        print(annFile)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)

class COCOForBatchLearning(CocoDetection):
    def __init__(self,train=True, transform=None,task_classes=None,task_id=None):
        if train:
            root = traindata_root
            ann_file=train_annFile
        else:
            root = valdata_root
            ann_file = val_annFile

        super().__init__(root=root,
                         annFile=ann_file,
                         transform=transform
                         )
        self.targets = [self.make_onehot(self.make_targets(self._load_target(id_))) for id_ in self.ids]
        self.c_nums = self.get_num_for_class()

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)

        # target = self._load_target(id)
        # target = self.make_targets(target)

        # target = self.make_onehot(self.targets[index])
        target = self.targets[index]
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def make_onehot(self, target):
        # len_onehot = len(coco_classes) + 1
        len_onehot = len(coco_classes)
        one_hot = torch.zeros([len_onehot])
        one_hot[target] = 1
        return one_hot

    def make_targets(self,target):
        class_ids = set()
        if target:
            for label in target:
                class_ids.add(label['category_id'])
            target = list(class_ids)
            target = [COCO_LABEL_MAP[e]-1 for e in target]
            # target.insert(0, 0)  # 保证在索引为0的位置，为background
        else:
            target = [0]
        return target

    def get_num_for_class(self):
        t = torch.stack(self.targets,0)
        res = {}
        cur_classes_tensor = torch.tensor(range(len(coco_classes)))
        for c in cur_classes_tensor:
            t_ = t[:,c]
            res[c] = [torch.sum(t_, 0), t_.shape[0] - torch.sum(t_, 0)]
        return res

class MultiLabelCOCO(CocoDetection):
    def __init__(self,train=True, transform=None,task_classes=None,task_id=None):
        if train:
            root = traindata_root
            ann_file=train_annFile
            self.imb_root = get_project_path_and_name()[0] + "saves/coco/imbalance/train/"
            self.pt_name = str(task_classes[task_id])+ ".pt"
        else:
            root = valdata_root
            ann_file = val_annFile
            self.imb_root = get_project_path_and_name()[0] + "saves/coco/imbalance/test/"
            self.pt_name = str(task_classes[task_id]) + ".pt"

        super().__init__(root=root,
                         annFile=ann_file,
                         transform=transform
                         )
        self.targets = [self.make_targets(self._load_target(id_)) for id_ in self.ids]

        # 找出所有带有目标类的data以及对应的target
        self.task_classes = task_classes
        self.task_id = task_id
        self.select_datas_for_classes()

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)

        # target = self._load_target(id)
        # target = self.make_targets(target)

        target = self.make_onehot(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def make_onehot(self, target):
        # len_onehot = len(coco_classes) + 1
        len_onehot = len(coco_classes)
        one_hot = torch.zeros([len_onehot])
        one_hot[target] = 1
        return one_hot

    def make_targets(self,target):
        class_ids = set()
        if target:
            for label in target:
                class_ids.add(label['category_id'])
            target = list(class_ids)
            target = [COCO_LABEL_MAP[e]-1 for e in target]
            # target.insert(0, 0)  # 保证在索引为0的位置，为background
        else:
            target = [0]
        return target

    def select_datas_for_classes(self):
        if self.task_classes is not None and self.task_id is not None:
            past_tasks = list(range(self.task_id)) if self.task_id > 0 else None
            self.seen_classes = []
            if past_tasks:
                self.seen_classes = [i for item in [self.task_classes[i] for i in past_tasks] for i in item]
            self.unseen_classes = list(set([i for item in self.task_classes for i in item]) - set(self.seen_classes) - set(
                self.task_classes[self.task_id]))

            self.current_classes = self.task_classes[self.task_id] # 10个类

            if_union = [True if len(set(self.targets[i]) & set(self.task_classes[self.task_id]))>0 else False for i in range(len(self.targets))]
            if_union = np.array(if_union)
            self.task_specfic_indexs= np.where(if_union==True)
            self.targets = [self.targets[i] for i in self.task_specfic_indexs[0]]

            # 只保留当前任务的类
            for i,target in enumerate(self.targets):
                self.targets[i] = list(set(target) & set(self.current_classes))

            self.ids = [self.ids[i] for i in self.task_specfic_indexs[0]]

            if not os.path.exists(self.imb_root+self.pt_name):
                self.reset_classes_to_imbalance(self.targets)

    def reset_classes_to_imbalance(self,targets):
        def cal_nums(target_array):
            res = {}
            cur_classes_tensor = torch.tensor(self.cur_classes)
            for c in cur_classes_tensor:
                t_ = target_array[:, c]
                res[c.item()] = torch.sum(t_, 0)
            return res

        target_array = torch.stack([self.make_onehot(t) for t in targets],0)
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
            print("distance:",d)
            if d == last_d:
                count = count+1
            else:
                count = 0
            last_d = d

        self.targets = [self.targets[i] for i in idx_list]
        self.ids = [self.ids[i] for i in idx_list]

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
            res[c] = torch.sum(t_,0)
        return res

class Imbalance_MultiLabelCOCO(CocoDetection):
    """持续学习中，每个任务都包含几个类，为了避免一个任务中的类别数量趋近于平衡，此函数类用于为每个任务
        都挑选不平衡的类别们"""
    def __init__(self,train=True, transform=None,task_classes=None,task_id=None,coco=None):
        if train:
            root = traindata_root
            ann_file=train_annFile
            self.imb_root = get_project_path_and_name()[0]+"saves/coco/imbalance/train/"
            self.pt_name = str(task_classes[task_id])+".pt"
        else:
            root = valdata_root
            ann_file = val_annFile
            self.imb_root = get_project_path_and_name()[0] + "saves/coco/imbalance/test/"
            self.pt_name = str(task_classes[task_id]) + ".pt"

        super().__init__(root=root,
                         annFile=ann_file,
                         transform=transform,
                         coco=coco
                         )
        if not os.path.exists(self.imb_root):
            os.makedirs(self.imb_root)
        self.targets = [self.make_targets(self._load_target(id_)) for id_ in self.ids]

        # 找出所有带有目标类的data以及对应的target
        self.task_classes = task_classes
        self.task_id = task_id
        self.select_datas_for_classes()
        self.c_nums = self.get_num_for_class()
        self.targets = [self.make_onehot(e) for e in self.targets]
        print(len(self.targets))

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)

        # target = self._load_target(id)
        # target = self.make_targets(target)

        # target = self.make_onehot(self.targets[index])
        target = self.targets[index]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def make_onehot(self, target):
        # len_onehot = len(coco_classes) + 1
        len_onehot = len(coco_classes)
        one_hot = torch.zeros([len_onehot])
        one_hot[target] = 1
        return one_hot

    def make_targets(self,target):
        class_ids = set()
        if target:
            for label in target:
                class_ids.add(label['category_id'])
            target = list(class_ids)
            target = [COCO_LABEL_MAP[e]-1 for e in target]
            # target.insert(0, 0)  # 保证在索引为0的位置，为background
        else:
            target = [0]
        return target

    def select_datas_for_classes(self):
        if self.task_classes is not None and self.task_id is not None:
            past_tasks = list(range(self.task_id)) if self.task_id > 0 else None
            self.seen_classes = []
            if past_tasks:
                self.seen_classes = [i for item in [self.task_classes[i] for i in past_tasks] for i in item]
            self.unseen_classes = list(set([i for item in self.task_classes for i in item]) - set(self.seen_classes) - set(
                self.task_classes[self.task_id]))

            self.cur_classes = self.task_classes[self.task_id] # 10个类

            if_union = [True if len(set(self.targets[i]) & set(self.task_classes[self.task_id]))>0 else False for i in range(len(self.targets))]
            if_union = torch.tensor(if_union)

            if os.path.exists(self.imb_root+self.pt_name):
                self.task_specfic_indexs = [torch.load(self.imb_root+self.pt_name)]
            else:
                self.task_specfic_indexs = torch.where(if_union==True) # 当前任务涉及到的样本索引

            self.targets = [self.targets[i] for i in self.task_specfic_indexs[0]]

            # 只保留当前任务的类
            for i,target in enumerate(self.targets):
                self.targets[i] = list(set(target) & set(self.cur_classes))

            self.ids = [self.ids[i] for i in self.task_specfic_indexs[0]]

            if not os.path.exists(self.imb_root+self.pt_name):
                print(self.imb_root+self.pt_name)
                self.reset_classes_to_imbalance(self.targets)

    def reset_classes_to_imbalance(self,targets):
        def cal_nums(target_array):
            res = {}
            cur_classes_tensor = torch.tensor(self.cur_classes)
            for c in cur_classes_tensor:
                t_ = target_array[:, c]
                res[c.item()] = torch.sum(t_, 0)
            return res

        target_array = torch.stack([self.make_onehot(t) for t in targets],0)
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
        self.ids = [self.ids[i] for i in idx_list]

        nums = [i.item() for i in tmp_res.values()]
        with open(self.imb_root+str(self.cur_classes)+".txt", 'w+', encoding='utf-8') as f:
            f.write(str(nums))

        sss = self.task_specfic_indexs[0][torch.tensor(idx_list)]
        torch.save(sss,self.imb_root+self.pt_name)

    def get_num_for_class(self):
        t = torch.stack([self.make_onehot(t) for t in self.targets], 0)
        res = {}
        cur_classes_tensor = torch.tensor(self.cur_classes)
        for c in cur_classes_tensor:
            t_ = t[:, c]
            res[c] = [torch.sum(t_, 0), t_.shape[0] - torch.sum(t_, 0)]
        return res

def collate_fn_coco(batch):
    b = list(zip(*batch))
    return tuple(zip(*batch))


def make_ml_coco_dataset():
    # 创建 coco dataset
    trans = T.Compose([T.ToTensor(), T.Resize([224, 224])])
    coco_train = MultiLabelCOCO(traindata_root, train_annFile, transform=trans)

    # 创建 coco sampler
    sampler = torch.utils.data.RandomSampler(coco_train)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=8, drop_last=True)

    # 创建 dataloader
    train_loader = torch.utils.data.DataLoader(
        coco_train,
        batch_sampler=batch_sampler,
        num_workers=2,
        # collate_fn=collate_fn_coco
    )
    return train_loader


def train():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = VisionTransformerB16(
        num_classes=len(coco_classes) + 1,
    )
    # model = vit_b_16(weights=ViT_B_16_Weights)
    model.to(device)

    data_loader = make_ml_coco_dataset()

    loss_fc = nn.BCELoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.01)

    # 可视化
    for imgs, labels in data_loader:
        optim.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)

        loss = loss_fc(out, labels)
        loss.backward()
        optim.step()

        print(loss)
        # print(out.shape)


def mv_unlabeled_imgs():
    with open("../nolabels.txt", mode="r") as f:
        img_ids = f.read().splitlines()
    img_names = []
    for id in img_ids:
        while len(id) != 12:
            id = '0' + id
        img_names.append(id + '.jpg')
    for img_name in img_names:
        shutil.move(traindata_root + img_name, dataset_root + "unlabeled_imgs/" + img_name)
        print(dataset_root + "unlabeled_imgs/" + img_name)


if __name__ == '__main__':
    train_dataset = Imbalance_MultiLabelCOCO(
        train=True,
        task_classes=[[0,5,8,9]],
        task_id=0)
