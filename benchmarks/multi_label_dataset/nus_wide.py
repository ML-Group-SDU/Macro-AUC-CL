import random

import torch
from torch.utils.data import Dataset
import scipy.io as sio
from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from tools.get_path import get_project_path_and_name,read_yaml

nus_root = read_yaml("paths.yaml")["nus_dir"]

class NUS_WIDE(Dataset):
    def __init__(self, train=True, transforms=None,task_classes = None,task_id = None):
        super().__init__()
        usr_root = os.path.expanduser("~")
        self.root = usr_root + nus_root
        if train:
            image_dir = self.root + '/ImageList/TrainImagelist.txt'
            mat_dir = self.root + "/mat/Train_labels.mat"
        else:
            image_dir = self.root + '/ImageList/TestImagelist.txt'
            mat_dir = self.root + "/mat/Test_labels.mat"

        label_dir = "AllLabels"
        txt_dir = 'NUS_WID_Tags/All_Tags.txt'
        output_dir = 'NUS_WIDE_10K/NUS_WIDE_10k.list'

        self.transforms = transforms

        with open(image_dir) as f:
            self.image_lists = f.readlines()

        labels_mat = loadmat(mat_dir)
        self.targets = labels_mat["labels"]


        self.task_classes = task_classes
        self.task_id = task_id
        self.select_datas_for_classes()
        self.c_nums = self.get_num_for_class()

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, index):
        image_path = self.image_lists[index].strip().split("\\")
        image_path = image_path[0]+"/"+image_path[1]

        img = Image.open(self.root + "/NUSWIDE/Flickr/" + image_path)
        if len(img.split()) != 3:
            img = img.convert('RGB')


        # target = torch.tensor(self.targets[index],dtype=torch.float)
        target = self.targets[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    def count_the_num_of_classes(self):
        counts = np.count_nonzero(self.targets,axis=0)
        print(counts)

    def select_datas_for_classes(self):
        if self.task_classes is not None and self.task_id is not None:
            past_tasks = list(range(self.task_id)) if self.task_id > 0 else None
            self.seen_classes = []
            if past_tasks:
                self.seen_classes = [i for item in [self.task_classes[i] for i in past_tasks] for i in item]
            self.unseen_classes = list(set([i for item in self.task_classes for i in item]) - set(self.seen_classes) - set(
                self.task_classes[self.task_id]))
            self.cur_classes = self.task_classes[self.task_id]

            target_array = torch.Tensor(self.targets)
            classes_array = target_array[:,self.cur_classes]
            res = torch.sum(classes_array,1)
            indexs = torch.where(res != 0)

            # 把没见过的类和过去类的信息去掉，只保留当前类信息
            target_array = target_array[indexs[0],:]
            non_current_classes = self.unseen_classes + self.seen_classes
            for id_ in non_current_classes:
                target_array[:,id_] = torch.zeros_like(target_array[:,id_])

            self.targets = target_array
            self.image_lists = [self.image_lists[i] for i in indexs[0]]

    def get_num_for_class(self):
        t = self.targets
        res = {}
        cur_classes_tensor = torch.tensor(self.cur_classes)
        for c in cur_classes_tensor:
            t_ = t[:,c]
            res[c] = torch.sum(t_,0)
        return res

class NUSForBatchLearning(Dataset):
    def __init__(self, train=True, transforms=None,task_classes = None,task_id = None):
        super().__init__()
        usr_root = os.path.expanduser("~")
        self.root = usr_root + nus_root
        if train:
            image_dir = self.root + '/ImageList/TrainImagelist.txt'
            mat_dir = self.root + "/mat/Train_labels.mat"
        else:
            image_dir = self.root + '/ImageList/TestImagelist.txt'
            mat_dir = self.root + "/mat/Test_labels.mat"

        label_dir = "AllLabels"
        txt_dir = 'NUS_WID_Tags/All_Tags.txt'
        output_dir = 'NUS_WIDE_10K/NUS_WIDE_10k.list'

        self.transforms = transforms

        with open(image_dir) as f:
            self.image_lists = f.readlines()

        labels_mat = loadmat(mat_dir)
        self.targets = labels_mat["labels"]
        self.targets = torch.Tensor(self.targets)
        self.c_nums = self.get_num_for_class()

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, index):
        image_path = self.image_lists[index].strip().split("\\")
        image_path = image_path[0]+"/"+image_path[1]

        img = Image.open(self.root + "/NUSWIDE/Flickr/" + image_path)
        if len(img.split()) != 3:
            img = img.convert('RGB')

        # target = torch.tensor(self.targets[index],dtype=torch.float)
        target = self.targets[index]

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    def count_the_num_of_classes(self):
        counts = np.count_nonzero(self.targets,axis=0)
        print(counts)

    def get_num_for_class(self):
        t = self.targets

        res = {}
        cur_classes_tensor = torch.tensor(range(t[0].shape[0]))
        for c in cur_classes_tensor:
            t_ = torch.tensor(t[:,c])
            print(t_)
            res[c] = [torch.sum(t_, 0), t_.shape[0] - torch.sum(t_, 0)]
        return res

class Imbalance_NUS_WIDE(Dataset):
    def __init__(self, train=True, transforms=None,task_classes = None,task_id = None):
        super().__init__()
        usr_root = os.path.expanduser("~")
        self.root = usr_root + nus_root
        if train:
            image_dir = self.root + '/ImageList/TrainImagelist.txt'
            mat_dir = self.root + "/mat/Train_labels.mat"
            self.imb_root = get_project_path_and_name()[0] + "saves/nus/imbalance/train/"
            if not os.path.exists(self.imb_root):
                os.makedirs(self.imb_root)
            self.pt_name = str(task_classes[task_id]) + ".pt"
        else:
            image_dir = self.root + '/ImageList/TestImagelist.txt'
            mat_dir = self.root + "/mat/Test_labels.mat"
            self.imb_root = get_project_path_and_name()[0] + "saves/nus/imbalance/test/"
            if not os.path.exists(self.imb_root):
                os.makedirs(self.imb_root)
            self.pt_name = str(task_classes[task_id]) + ".pt"

        label_dir = "AllLabels"
        txt_dir = 'NUS_WID_Tags/All_Tags.txt'
        output_dir = 'NUS_WIDE_10K/NUS_WIDE_10k.list'

        self.transforms = transforms

        with open(image_dir) as f:
            self.image_lists = f.readlines()

        labels_mat = loadmat(mat_dir)
        self.targets = labels_mat["labels"]


        self.task_classes = task_classes
        self.task_id = task_id
        self.select_datas_for_classes()
        self.c_nums = self.get_num_for_class()

        self.targets = [self.targets[i] for i in range(self.targets.shape[0])]
        print(len(self.targets))

    def __len__(self):
        return len(self.image_lists)

    def __getitem__(self, index):
        image_path = self.image_lists[index].strip().split("\\")
        image_path = image_path[0]+"/"+image_path[1]

        img = Image.open(self.root + "/NUSWIDE/Flickr/" + image_path)
        if len(img.split()) != 3:
            img = img.convert('RGB')


        # target = torch.tensor(self.targets[index],dtype=torch.float)
        target = self.targets[index]

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def count_the_num_of_classes(self):
        counts = np.count_nonzero(self.targets,axis=0)
        print(counts)

    def select_datas_for_classes(self):
        if self.task_classes is not None and self.task_id is not None:
            past_tasks = list(range(self.task_id)) if self.task_id > 0 else None
            self.seen_classes = []
            if past_tasks:
                self.seen_classes = [i for item in [self.task_classes[i] for i in past_tasks] for i in item]
            self.unseen_classes = list(set([i for item in self.task_classes for i in item]) - set(self.seen_classes) - set(
                self.task_classes[self.task_id]))
            self.cur_classes = self.task_classes[self.task_id]

            target_array = torch.Tensor(self.targets)
            classes_array = target_array[:,self.cur_classes]
            res = torch.sum(classes_array,1)

            if os.path.exists(self.imb_root + self.pt_name):
                self.task_specfic_indexs = [torch.load(self.imb_root + self.pt_name)]
            else:
                self.task_specfic_indexs = torch.where(res != 0)  # 当前任务涉及到的样本索引

            # 把没见过的类和过去类的信息去掉，只保留当前类信息
            target_array = target_array[self.task_specfic_indexs[0],:]
            non_current_classes = self.unseen_classes + self.seen_classes
            for id_ in non_current_classes:
                target_array[:,id_] = torch.zeros_like(target_array[:,id_])

            self.targets = target_array
            self.image_lists = [self.image_lists[i] for i in self.task_specfic_indexs[0]]

            if not os.path.exists(self.imb_root + self.pt_name):
                self.reset_classes_to_imbalance(target_array)

    def get_num_for_class(self):
        t = self.targets
        res = {}
        cur_classes_tensor = torch.tensor(self.cur_classes)
        for c in cur_classes_tensor:
            t_ = t[:, c]
            res[c] = [torch.sum(t_, 0), t_.shape[0] - torch.sum(t_, 0)]
        return res

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

        self.targets = target_array[torch.tensor(idx_list),:]
        self.image_lists = [self.image_lists[i] for i in idx_list]

        nums = [i.item() for i in tmp_res.values()]
        with open(self.imb_root+str(self.cur_classes)+".txt", 'w+', encoding='utf-8') as f:
            f.write(str(nums))

        sss = self.task_specfic_indexs[0][torch.tensor(idx_list)]
        torch.save(sss,self.imb_root+self.pt_name)

def make_one_hot_labels():
    root_ = "/home/zhangyan/data/Datasets/NUS-WIDE/"

    print("--- label ---")
    LABEL_P = root_ + "/TrainTestLabels"

    # class order determined by `Concepts81.txt`
    cls_id = {}
    with open(root_+"Concepts81.txt", "r") as f:
        for cid, line in enumerate(f):
            cn = line.strip()
            cls_id[cn] = cid
    # print("\nclass-ID:", cls_id)
    id_cls = {cls_id[k]: k for k in cls_id}
    # print("\nID-class:", id_cls)
    N_CLASS = len(cls_id)
    print("\n#classes:", N_CLASS)

    train_class_files,test_class_files = [],[]
    for filename in os.listdir(LABEL_P):
        if "Train" in filename:
            train_class_files.append(filename)
        elif "Test" in filename:
            test_class_files.append(filename)

    def aaa(class_files,mode):

        with open(root_+f"ImageList/{mode}Imagelist.txt") as f:
            N_SAMPLE = len(f.readlines())

        # print("\nlabel file:", len(class_files), class_files)
        label_key = lambda x: x.split(".txt")[0].split("_")[1]

        labels = np.zeros([N_SAMPLE, N_CLASS], dtype=np.int8)
        for cf in class_files:
            c_name = label_key(cf)
            cid = cls_id[c_name]
            print('->', cid, c_name)
            with open(os.path.join(LABEL_P, cf), "r") as f:
                for sid, line in enumerate(f):
                    if int(line) > 0:
                        labels[sid][cid] = 1
        print("labels:", labels.shape, ", cardinality:", labels.sum())
        # labels: (269648, 81) , cardinality: 503848
        # np.save("labels.npy", labels.astype(np.int8))
        labels = labels.astype(np.uint8)
        sio.savemat(root_ + f"/mat/{mode}_labels.mat", {"labels": labels}, do_compression=True)

    aaa(test_class_files,"Test")



if __name__ == '__main__':

    train_dataset = Imbalance_NUS_WIDE(
        train=True,
        task_classes=[[2,3,4,5],[6,7,8,9]],
        task_id=0)