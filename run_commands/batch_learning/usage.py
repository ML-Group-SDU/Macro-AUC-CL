import os
import time

from run_commands.MyClock import MyClock
from tools.get_path import get_project_path, get_log_path

py3 = "~/anaconda3/envs/ava/bin/python "


def naive_usage(des,dataset_name, crit, py_file,
                seed_list=[2222], epoch=18, batch_size=128,
                opt="adam", cu=3, lr=0.0005, Cs=None, nth_power=1 / 4, num_workers=0,eval_every=0
                ):
    if Cs is None:
        Cs = [0.0]

    cmds = []
    for seed in seed_list:
        for C in Cs:
            log = get_log_path() + f"/multilabel_batch_learning/{dataset_name}/{des}{crit}-{opt}-C{C}/e{epoch}-s{seed}-b{batch_size}"

            template_g = py3 + \
                         get_project_path() + f"/examples/multi_label/{py_file}" + \
                         f" --dataset_name={dataset_name} --crit={crit} --opt={opt} " + \
                         f"--cuda={cu} --lr={lr} --batch_size={batch_size} " + \
                         (f"--seed={seed} --epoch={epoch} --csvlog_dir={log} --C={C} --nth_power={nth_power} "
                          f"--num_workers={num_workers} --eval_every={eval_every}")

            cmds.append(template_g)

    mc = MyClock()

    for c in cmds:
        os.system(c)
        mc.cal_time()

def ddp_run():

    command = py3 + "-m torch.distributed.run --nproc_per_node=2 "
    script = "/home2/zhangyan/codes/avalanche/examples/multi_label/multi_label_batch_learning.py --dataset_name=coco --batch_size=256"
    os.system(command+script)

if __name__ == '__main__':
    ddp_run()