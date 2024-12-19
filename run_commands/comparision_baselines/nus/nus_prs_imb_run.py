import sys

sys.path.insert(0, "/home2/zhangyan/codes/MACRO-AUC-CL/")

import run_commands.multi_label.usage as usa
from run_commands.dataset_setup import *

setup = nus_setup

if __name__ == '__main__':
    usa.run_ml_cl_tasks_from_single_dataset(
        dataset_name=setup['dataset'],
        task_num=setup['task_num'],
        des="prsmap",
        crit="MultiSoft",
        measure='map',
        py_file="multi_label_cl/comparison/prs_imbalance.py",
        seed_list=[2222],
        epoch=40,
        batch_size=512,
        opt="sgd",
        cu=setup["cuda"],
        lr=0.01,
        use_aug="no",
        mem_sizes=[2000],
        model="res34",
        imbalance="yes",
        num_workers=2
    )
