from run_commands.batch_learning.usage import naive_usage
common_args = {
    'batch_size':192,
    'learning_rate':0.005,
    'num_epochs':30,
    'cuda':2,
    'num_workers':2
}
if __name__ == '__main__':
    naive_usage(
        des="2step",
        dataset_name="voc",
        crit="2step",
        py_file="multi_label_batch_learning_2step.py",
        seed_list=[2222],
        epoch=common_args["num_epochs"],
        batch_size=common_args["batch_size"],
        opt="sgd",
        cu=common_args['cuda'],
        lr=common_args['learning_rate'],
        num_workers=common_args['num_workers'],
        Cs=[0.7,3]
    )