import sys
sys.path.insert(0, "/home/zhangyan/codes/MACRO-AUC-CL/")

import run_commands.multi_label.usage as usa
from run_commands.dataset_setup import *

setup = nus_setup

if __name__ == '__main__':
    usa.run_ml_cl_tasks_from_single_dataset(
        dataset_name=setup['dataset'],
        task_num=setup['task_num'],
        des="rebuttal_nips",
        crit="RM",
        measure='macro-auc',
        py_file="multi_label_cl/rm_wrs.py",
        seed_list=[2222],
        epoch=40,
        batch_size=256,
        opt="sgd",
        cu=setup['cuda'],
        lr=0.01,
        model="res34",
        imbalance="yes",
        mem_sizes=[2000],
        lambdas=[1.2],
        num_workers=2,
        scale_ratio_list=[1.0]
    )