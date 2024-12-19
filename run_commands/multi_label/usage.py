import os

from run_commands.MyClock import MyClock
from tools.get_path import get_project_path_and_name, get_log_path, get_python_exec_path
import itertools

py3 = get_python_exec_path("/anaconda3/envs/ava/bin/python ")

def naive_usage(dataset_name, crit, py_file,
                seed_list=[2222], epoch=18, batch_size=128, opt="adam", cu=3, lr=0.0005,
                ):
    cmds = []
    for seed in seed_list:
        log = get_log_path() + f"/train_logs/multilabel/{dataset_name}/{crit}-{opt}/e{epoch}-s{seed}-b{batch_size}"

        template_g = py3 + \
                     get_project_path_and_name()[0] + f"/examples/macro-auc/{py_file}" + \
                     f" --dataset_name={dataset_name} --crit={crit} --opt={opt} " \
                     f"--cuda={cu} --lr={lr} --batch_size={batch_size} --seed={seed} --epoch={epoch} --csvlog_dir={log}"

        cmds.append(template_g)

    mc = MyClock()

    for c in cmds:
        os.system(c)
        mc.cal_time()

def binary_classification_usage(crit, py_file, seed_list=[2222], epoch=18, batch_size=128, opt="adam", cu=3, lr=0.0005,
                                two_class_id=None,train_scalers=None,test_scalers=None,weight_decay=None):
    cmds = []
    for seed in seed_list:
        log = get_log_path() + f"/train_logs/binary_class/{two_class_id[0]}_{two_class_id[1]}-{train_scalers}-{test_scalers}/{crit}-{opt}/w{weight_decay}-e{epoch}-s{seed}-b{batch_size}"

        template_g = py3 + \
                     get_project_path_and_name()[0] + f"/examples/macro-auc/{py_file}" + \
                     f" --crit={crit} --opt={opt} --cuda={cu} --lr={lr} --batch_size={batch_size}" \
                     f" --seed={seed} --epoch={epoch} --csvlog_dir={log} --weight_decay={weight_decay}" \
                     f" --class_id1={two_class_id[0]} --class_id2={two_class_id[1]}" \
                     f" --train_scalers={train_scalers} --test_scalers={test_scalers}"

        cmds.append(template_g)

    mc = MyClock()

    for c in cmds:
        os.system(c)
        mc.cal_time()


def run_ml_cl_tasks_from_single_dataset(des,dataset_name,task_num, crit,measure,
                                 py_file,seed_list=[2222], epoch=30, batch_size=128, opt="adam", cu=3, lr=0.0005,
                                 use_aug="no",mem_sizes=[2000],imbalance="yes",model="res50",lambdas=[0], num_workers=0,
                                 with_wrs="no",scale_ratio_list=[1.0]
                ):
    cmds = []
    for seed, lam, mem_size,sr in itertools.product(seed_list, lambdas, mem_sizes,scale_ratio_list):
        log = get_log_path() + f"/train_logs/MLCL/{dataset_name}/{task_num}task,mem{mem_size},C{lam},{des}/{crit}-{measure}-{opt}/e{epoch}-s{seed}-b{batch_size}-sr{sr}"
        template_g = py3 + \
                     get_project_path_and_name()[0] + f"/train/{py_file}" + \
                     f" --dataset_name={dataset_name} --task_num={task_num} --crit={crit} --measure={measure} --opt={opt}" \
                     f" --cuda={cu} --lr={lr} --batch_size={batch_size} --seed={seed} --epoch={epoch} --csvlog_dir={log}" \
                     f" --use_aug={use_aug} --mem_size={mem_size} --imbalance={imbalance} --model={model} --C={lam} --num_workers={num_workers}" \
                     f" --with_wrs={with_wrs} --scale_ratio={sr}"
        cmds.append(template_g)

    mc = MyClock()

    for cmd in cmds:
        os.system(cmd)
        mc.cal_time()

def ablation_usage(des,dataset_name,task_num, crit,measure,
                                 py_file,seed_list=[2222], epoch=30, batch_size=128, opt="adam", cu=3, lr=0.0005,
                                 use_aug="no",mem_sizes=[2000],imbalance="yes",model="res50",cs=[0], num_workers=0,
                                 with_wrs="no",memory_update="wrs"
                ):
    cmds = []
    for seed in seed_list:
        for C in cs:
            for mem_size in mem_sizes:
                log = get_log_path() + f"/train_logs/ablation/{dataset_name}/{task_num}task,mem{mem_size},C{C},{des}/{crit}-{measure}-{opt}/e{epoch}-s{seed}-b{batch_size}"

                template_g = py3 + \
                             get_project_path_and_name()[0] + f"/examples/multi_label/ablation/{py_file}" + \
                             f" --dataset_name={dataset_name} --task_num={task_num} --crit={crit} --measure={measure} --opt={opt} " \
                             f"--cuda={cu} --lr={lr} --batch_size={batch_size} --seed={seed} --epoch={epoch} --csvlog_dir={log}" \
                             f" --use_aug={use_aug} --mem_size={mem_size} --imbalance={imbalance} --model={model} --C={C} --num_workers={num_workers}" \
                             f" --with_wrs={with_wrs} --memory_update={memory_update}"

                cmds.append(template_g)

    mc = MyClock()

    for c in cmds:
        os.system(c)
        mc.cal_time()

__ALL__={
    "run_ml_cl_tasks_from_single_dataset",
}