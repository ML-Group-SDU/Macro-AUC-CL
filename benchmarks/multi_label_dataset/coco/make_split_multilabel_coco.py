""""
基于COCO数据集，制作Continual learning中的Multi-Label Benchmark；
"""""
from torchvision.datasets import CocoDetection
import os
import torchvision.transforms as T
from tools.coco.coco_classes import *


# %%
class COCO_ID_TAR(CocoDetection):
    def __getitem__(self, index):
        id = self.ids[index]
        target = self._load_target(id)

        class_ids = set()
        if target:
            for label in target:
                class_ids.add(label['category_id'])
            target = list(class_ids)
            target = [COCO_LABEL_MAP[e] for e in target]
            target.insert(0, 0)
        else:
            target = [0]

        return id, target


# %%
dataset_root = "/mnt/d/forlinux/dataset/coco2017/"
root = dataset_root + 'train/data/'
annFile = dataset_root + 'annotations/instances_train2017.json'

trans = T.Compose([T.ToTensor(), T.Resize([224, 224])])
coco = COCO_ID_TAR(root, annFile, transform=trans)

# %%
dir_root = "/mnt/d/forlinux/codes/avalanche/zy_tools/files/"
for file in os.listdir(dir_root):
    os.remove(dir_root+file)

#%%
for i in range(len(coco)):
    id_ = coco[i][0]
    targets = coco[i][-1]

    # 获得图片文件名
    id_ = str(id_)
    while len(id_) != 12:
        id_ = '0' + id_
    img_name = id_ + ".jpg"

    # 获取图片中包含的类别
    targets_str = ''
    for s in targets:
        targets_str += (str(s)+",")

    # 对于每一个类别，都在该类别对应的文件中写入一条记录
    for target in targets:
        with open(dir_root + str(target) + ".txt", mode="a+") as f:
            f.write(img_name + " " + targets_str + "\n")



# %%
anno = coco[0][1]
for a in anno:
    print(a.keys())
