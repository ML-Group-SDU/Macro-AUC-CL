from run_commands.multi_label.usage import ablation_usage
commom_setup={
    'cuda':'0'
}
voc_setup = {
    'dataset':'voc',
    'task_num':5,

}
coco_setup = {
    'dataset':'coco',
    'task_num':8,
    'cuda':'2'
}
nus_setup = {
    'dataset':'nus',
    'task_num':9,
    'cuda':'2'
}
setup = nus_setup

if __name__ == '__main__':
    ablation_usage(
        des="only_rm_no_wrs",
        dataset_name=setup['dataset'],
        task_num=setup['task_num'],
        crit="RM",
        measure='macro-auc',
        py_file="rm_ablation.py",
        seed_list=[2222],
        epoch=40,
        batch_size=128,
        opt="sgd",
        cu=commom_setup['cuda'],
        lr=0.01,
        model="res34",
        imbalance="yes",
        mem_sizes=[2000],
        cs=[3.5],
        num_workers=2,
        with_wrs="no"
    )