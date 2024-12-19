################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 21-04-2022                                                             #
# Author(s): Antonio Carta, Lorenzo Pellegrini                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import itertools
import random
from collections import defaultdict

import torch
import numpy as np


def get_weighted_mem(mem, similarity_order, mu):
    list1 = [round(mu ** i, 2) for i in range(len(similarity_order))]
    task_w_ratio = [round((a / sum(list1)) * len(similarity_order), 2) for a in list1]
    for i, v in enumerate(similarity_order):
        idx = torch.where(mem[-1] == v)
        mem[0][idx] = mem[0][idx] * task_w_ratio[i]
        mem[2][idx] = mem[2][idx] * task_w_ratio[i]
    return mem


def classification_collate_mbatches_fn(mbatches, similarity_order=None, mu=None):
    """Combines multiple mini-batches together.

    Concatenates each tensor in the mini-batches along dimension 0 (usually
    this is the batch size).

    :param mbatches: sequence of mini-batches.
    :return: a single mini-batch
    """
    batch = []
    for i in range(len(mbatches[0])):
        t = classification_single_values_collate_fn(
            [el[i] for el in mbatches], i
        )
        batch.append(t)
    return batch


def classification_single_values_collate_fn(values_list, index):
    """
    Collate function used to merge the single elements (x or y or t,
    etcetera) of a minibatch of data from a classification dataset.

    This function assumes that all values are tensors of the same shape
    (excluding the first dimension).

    :param values_list: The list of values to merge.
    :param index: The index of the element. 0 for x values, 1 for y values,
        etcetera. In this implementation, this parameter is ignored.
    :return: The merged values.
    """
    return torch.cat(values_list, dim=0)


def deal_list_to_module_tensor(mbatches):
    batch = []
    for i in range(len(mbatches[0])):
        values_list = [el[i] for el in mbatches]
        if isinstance(values_list[0], torch.Tensor):
            t = torch.stack(values_list, dim=0)
        elif isinstance(values_list[0], int):
            t = torch.tensor(values_list)
        batch.append(t)
    return batch


def mixup_classification_collate_mbatches_fn(mbatches, mix_mode, mix_times, nodes_num=10, scenario_mode='', similarity_order=None, mu=None):
    extend_ = False
    append_ = True
    if extend_:
        """将mabatchs里的数据，按照任务分开"""
        task_ids_set = list(set([item[-1] for item in mbatches]))
        task_ids_set.sort(reverse=True)  # 降序
        datas = []
        targets = []
        taskids = []
        for item in mbatches:
            datas.append(item[0])
            targets.append(item[1])
            taskids.append(item[2])

        datas = torch.stack(datas, 0)
        targets = torch.tensor(targets)
        taskids = torch.tensor(taskids)

        current_batchs = []
        memory_batchs = []

        current_t_id = task_ids_set[0]
        current_task_idx = torch.where(taskids == current_t_id)
        current_batchs = [
            datas[current_task_idx],
            targets[current_task_idx],
            taskids[current_task_idx]
        ]

        mem_task_idx = torch.where(taskids != current_t_id)
        memory_batchs = [
            datas[mem_task_idx],
            targets[mem_task_idx],
            taskids[mem_task_idx]
        ]

        # for i in range(len(task_ids_set)):
        #     task_id = task_ids_set[i]
        #     task_idx = torch.where(taskids == task_id)
        #     this_task_subset = [
        #         datas[task_idx],
        #         targets[task_idx],
        #         taskids[task_idx]
        #     ]
        #     if i == 0:
        #         current_batchs.extend(this_task_subset)
        #     else:
        #         memory_batchs.append(this_task_subset)

        mbatches = [current_batchs, memory_batchs]

    elif append_:
        for i in range(len(mbatches)):
            mbatches[i] = deal_list_to_module_tensor(mbatches[i])

        if len(mbatches) > 2:
            mem_tasks = mbatches[1:]
            datas = []
            targets = []
            taskids = []
            for mem_single_task in mem_tasks:
                datas.append(mem_single_task[0])
                targets.append(mem_single_task[1])
                taskids.append(mem_single_task[2])

            datas = torch.cat(datas, 0)
            targets = torch.cat(targets, 0)
            taskids = torch.cat(taskids, 0)

            mbatches = [
                mbatches[0],
                [datas, targets, taskids]
            ]

    if scenario_mode == 'task':
        return task_il(mbatches,mix_mode, mix_times, nodes_num, similarity_order, mu)
    elif scenario_mode == 'class':
        return class_il(mbatches,mix_mode, mix_times, nodes_num, similarity_order, mu)
    elif scenario_mode == "domain":
        return class_il(mbatches,mix_mode, mix_times, nodes_num, similarity_order, mu)
    else:
        print("Scenario Mode Error!!!!!!!!")


def traditional_mixup(mbatches, mixtimes, class_num, with_ori_data=True):
    index_list = list(range(len(mbatches)))
    random.shuffle(index_list)
    lam = get_lam()
    mixed_batches = []

    y_s = [item[1] for item in mbatches]
    onehots = make_onehots(class_num, y_s)
    for i in range(len(mbatches)):
        mbatches[i][1] = onehots[i]

    sample_len = int(mixtimes*len(mbatches))

    sample_list = list(enumerate(index_list))
    sample_list = random.choices(sample_list, k=sample_len)

    for k, v in sample_list:
        x = lam*mbatches[k][0] + (1-lam)*mbatches[v][0]
        y = lam*onehots[k] + (1-lam)*onehots[v]
        t = mbatches[k][2]
        mixed_batches.append([x, y, t])

    if with_ori_data:
        mbatches.extend(mixed_batches)
    else:
        mbatches = mixed_batches

    mbatches = deal_list_to_module_tensor(mbatches)
    return mbatches


def make_onehots(class_num, targets):
    one_hots = [torch.zeros(class_num) for _ in range(len(targets))]
    for i in range(len(targets)):
        one_hots[i][targets[i]] = 1
    return one_hots

def get_lam():
    lam = round(np.random.beta(1.0, 1.0), 2)
    return lam

def class_il(mbatches,mix_mode, mix_times, nodes_num=10, similarity_order=None, mu=None):
    if len(mbatches) == 2:  # 做 mixup
        data = mbatches[0]
        mem = mbatches[1]

        mixed_data = []
        mixed_targets = []
        mixed_task_ids = []
        mixed_one_hots = []

        len_d = data[0].shape[0]
        len_m = mem[0].shape[0]

        one_hots_d = [torch.zeros(nodes_num) for _ in range(len_d)]
        one_hots_m = [torch.zeros(nodes_num) for _ in range(len_m)]

        for i in range(len_d):
            one_hots_d[i][data[1][i]] = 1
        for j in range(len_m):
            one_hots_m[j][mem[1][j]] = 1

        mbatches[0].insert(2, torch.stack(one_hots_d, 0))
        mbatches[1].insert(2, torch.stack(one_hots_m, 0))

        # 做加权
        if (similarity_order is not None) and (mu is not None) and (mu > 0):
            mem = get_weighted_mem(mem, similarity_order, mu)
            mbatches[1] = mem
        def get_mixed_data(data1,len1,data2,len2,one_hots1,one_hots2):
            index1 = random.randint(0, len1 - 1)
            index2 = random.randint(0, len2 - 1)
            img1, target1, task1 = data1[0][index1], data1[1][index1], data1[-1][index1]
            img2, target2, task2 = data2[0][index2], data2[1][index2], data2[-1][index2]
            assert task1 == task2
            lam = get_lam()
            mixed_img = lam * img1 + (1 - lam) * img2
            mixed_one_hot = lam * one_hots1[index1] + (1 - lam) * one_hots2[index2]
            return mixed_img,target1,mixed_one_hot,task1

        mix_len = int(len_d * mix_times) if mix_mode == "d_mix" or "cross_mix" else int(len_m * mix_times)
        if mix_len > 0 and mix_mode != "no":
            for _ in range(mix_len):
                # d_index = random.randint(0, len_d - 1)
                # m_index = random.randint(0, len_m - 1)
                # d_img, d_target, d_task = data[0][d_index], data[1][d_index], data[-1][d_index]
                # m_img, m_target, m_task = mem[0][m_index], mem[1][m_index], mem[-1][m_index]
                #
                # lam = get_lam()
                # mixed_img = lam * d_img + (1 - lam) * m_img
                # mixed_one_hot = lam * one_hots_d[d_index] + (1 - lam) * one_hots_m[m_index]
                if mix_mode == "cross_mix":
                    mixed_img,mixed_target,mixed_one_hot,mixed_task = get_mixed_data(data,len_d,mem,len_m,one_hots_d,one_hots_m)
                elif mix_mode == "d_mix":
                    mixed_img, mixed_target, mixed_one_hot, mixed_task = get_mixed_data(data, len_d, data, len_d,
                                                                                        one_hots_d, one_hots_d)
                elif mix_mode == "m_mix":
                    mixed_img, mixed_target, mixed_one_hot, mixed_task = get_mixed_data(mem, len_m, mem, len_m,
                                                                                        one_hots_m, one_hots_m)
                else:
                    raise Exception
                mixed_data.extend([mixed_img])
                mixed_targets.extend([mixed_target])
                mixed_one_hots.extend([mixed_one_hot])
                mixed_task_ids.extend([mixed_task])

            mixed_data = torch.stack(mixed_data, 0)
            mixed_targets = torch.stack(mixed_targets, 0)
            mixed_one_hots = torch.stack(mixed_one_hots, 0)
            mixed_task_ids = torch.stack(mixed_task_ids, 0)

            mbatches.append([mixed_data, mixed_targets, mixed_one_hots, mixed_task_ids])

        res = classification_collate_mbatches_fn(mbatches)

    elif len(mbatches) == 1:  # 制作onehot
        data = mbatches[0]
        targets = data[1]
        one_hots = [torch.zeros(nodes_num) for _ in range(targets.shape[0])]
        for i in range(targets.shape[0]):
            one_hots[i][targets[i]] = 1
        mbatches[0].insert(2, torch.stack(one_hots))
        res = classification_collate_mbatches_fn(mbatches)

    return res


def task_il(mbatches,mix_mode, mix_times, nodes_num=10, similarity_order=None, mu=None):
    if len(mbatches) == 2:  # 做 mixup
        data = mbatches[0]
        mem = mbatches[1]

        mixed_data = []
        mixed_targets = []
        mixed_task_ids = []
        mixed_one_hots = []

        len_d = data[0].shape[0]
        len_m = mem[0].shape[0]

        one_hots_d = [torch.zeros(nodes_num) for _ in range(len_d)]
        one_hots_m = [torch.zeros(nodes_num) for _ in range(len_m)]

        for i in range(len_d):
            one_hots_d[i][data[1][i]] = 1
        for j in range(len_m):
            one_hots_m[j][mem[1][j]] = 1

        mbatches[0].insert(2, torch.stack(one_hots_d, 0))
        mbatches[1].insert(2, torch.stack(one_hots_m, 0))

        # 做加权
        if (similarity_order is not None) and (mu > 0):
            mem = get_weighted_mem(mem, similarity_order, mu)
            mbatches[1] = mem

        if mix_times > 0:
            if mix_mode == "cross_mix":
                for _ in range(int(len_m * mix_times)):
                    d_index = random.randint(0, len_d - 1)
                    m_index = random.randint(0, len_m - 1)
                    d_img, d_target, d_task = data[0][d_index], data[1][d_index], data[-1][d_index]
                    m_img, m_target, m_task = mem[0][m_index], mem[1][m_index], mem[-1][m_index]

                    lam = round(np.random.beta(0.5, 0.2), 2)
                    mixed_img = lam * d_img + (1 - lam) * m_img
                    # 这里不能简单的mixup了
                    mixed_one_hot = [
                        lam * one_hots_d[d_index],
                        (1 - lam) * one_hots_m[m_index]
                    ]

                    # 全部送
                    mixed_data.extend([mixed_img, mixed_img])
                    mixed_targets.extend([d_target, m_target])
                    mixed_one_hots.extend([mixed_one_hot[0], mixed_one_hot[1]])
                    mixed_task_ids.extend([d_task, m_task])

                    # 只送过去任务
                    # mixed_data.extend([mixed_img])
                    # mixed_targets.extend([m_target])
                    # mixed_one_hots.extend([mixed_one_hot[1]])
                    # mixed_task_ids.extend([m_task])

                    # 只送当前任务
                    # mixed_data.extend([mixed_img])
                    # mixed_targets.extend([d_target])
                    # mixed_one_hots.extend([mixed_one_hot[0]])
                    # mixed_task_ids.extend([d_task])

            elif mix_mode == "m_mix":
                lam = round(np.random.beta(0.2, 0.2), 2)
                index = torch.randperm(len_m).cuda()
                if mix_times <= 1:
                    index = index[:int(mix_times * len_m)]
                else:
                    index = index + index[:int((mix_times - 1) * len_m)]

                for k, v in enumerate(index):
                    m_img, m_target, m_one_hot, m_task = mem[0][k], mem[1][k], one_hots_m[k], mem[-1][k]
                    rnd_m_img, rnd_m_target, rnd_m_one_hot, rnd_m_task = mem[0][v], mem[1][v], one_hots_m[v], mem[-1][v]

                    mixed_img = m_img * lam + rnd_m_img * (1 - lam)
                    if m_task == rnd_m_task:
                        mixed_one_hot = lam * m_one_hot + (1 - lam) * rnd_m_one_hot

                        mixed_data.extend([mixed_img])
                        mixed_targets.extend([m_target])
                        mixed_one_hots.extend([mixed_one_hot])
                        mixed_task_ids.extend([m_task])
                    else:
                        mixed_one_hot = [lam * m_one_hot, (1 - lam) * rnd_m_one_hot]

                        mixed_data.extend([mixed_img, mixed_img])
                        mixed_targets.extend([m_target, rnd_m_target])
                        mixed_one_hots.extend([mixed_one_hot[0], mixed_one_hot[1]])
                        mixed_task_ids.extend([m_task, rnd_m_task])

            elif mix_mode == "d_mix":
                lam = round(np.random.beta(0.5, 0.2), 2)
                index = torch.randperm(len_d).cuda()

                if mix_times <= 1:
                    index = index[:int(mix_times * len_d)]
                else:
                    index = index + index[:int((mix_times - 1) * len_d)]

                for k, v in enumerate(index):
                    d_img, d_target, d_one_hot, d_task = data[0][k], data[1][k], one_hots_d[k], data[-1][k]
                    rnd_d_img, rnd_d_target, rnd_d_one_hot, rnd_d_task = data[0][v], data[1][v], one_hots_d[v], \
                    data[-1][v]

                    mixed_img = d_img * lam + rnd_d_img * (1 - lam)
                    mixed_one_hot = lam * d_one_hot + (1 - lam) * rnd_d_one_hot

                    mixed_data.extend([mixed_img])
                    mixed_targets.extend([d_target])
                    mixed_one_hots.extend([mixed_one_hot])
                    mixed_task_ids.extend([d_task])

            else:
                print("No right mixup")

            mixed_data = torch.stack(mixed_data, 0)
            mixed_targets = torch.stack(mixed_targets, 0)
            mixed_one_hots = torch.stack(mixed_one_hots, 0)
            mixed_task_ids = torch.stack(mixed_task_ids, 0)

            mbatches.append([mixed_data, mixed_targets, mixed_one_hots, mixed_task_ids])

        res = classification_collate_mbatches_fn(mbatches)

    elif len(mbatches) == 1:  # 制作onehot
        data = mbatches[0]
        targets = data[1]
        one_hots = [torch.zeros(nodes_num) for _ in range(targets.shape[0])]
        for i in range(targets.shape[0]):
            one_hots[i][targets[i]] = 1
        mbatches[0].insert(2, torch.stack(one_hots))
        res = classification_collate_mbatches_fn(mbatches)

    return res


def detection_collate_fn(batch):
    """
    Collate function used when loading detection datasets using a DataLoader.

    This will merge the single samples of a batch to create a minibatch.
    This collate function follows the torchvision format for detection tasks.
    """
    return tuple(zip(*batch))


def detection_collate_mbatches_fn(mbatches):
    """
    Collate function used when loading detection datasets using a DataLoader.

    This will merge multiple batches to create a concatenated batch.

    Beware that merging multiple batches is different from creating a batch
    from single dataset elements: Batches can be created from a
    list of single dataset elements by using :func:`detection_collate_fn`.
    """
    lists_dict = defaultdict(list)
    for mb in mbatches:
        for mb_elem_idx, mb_elem in enumerate(mb):
            lists_dict[mb_elem_idx].append(mb_elem)

    lists = []
    for mb_elem_idx in range(max(lists_dict.keys()) + 1):
        lists.append(
            list(itertools.chain.from_iterable(lists_dict[mb_elem_idx]))
        )

    return lists


__all__ = [
    'classification_collate_mbatches_fn',
    'classification_single_values_collate_fn',
    'mixup_classification_collate_mbatches_fn',
    'detection_collate_fn',
    'detection_collate_mbatches_fn'
]
