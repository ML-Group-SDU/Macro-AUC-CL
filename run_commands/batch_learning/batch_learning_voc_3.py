from run_commands.batch_learning.usage import naive_usage
common_args = {
    'batch_size':192,
    'learning_rate':0.005,
    'num_epochs':50,
    'cuda':2,
    'num_workers':2
}
if __name__ == '__main__':
    naive_usage(
        des = "eval5/",
        dataset_name="voc",
        crit="R",
        py_file="multi_label_batch_learning.py",
        seed_list=[156,0],
        epoch=common_args["num_epochs"],
        batch_size=common_args["batch_size"],
        opt="sgd",
        cu=common_args['cuda'],
        lr=common_args['learning_rate'],
        num_workers=common_args['num_workers'],
        eval_every=5,
        Cs=[1,2,0]
    )
    naive_usage(
        des="eval0/",
        dataset_name="voc",
        crit="R",
        py_file="multi_label_batch_learning.py",
        seed_list=[156, 0],
        epoch=common_args["num_epochs"],
        batch_size=common_args["batch_size"],
        opt="sgd",
        cu=common_args['cuda'],
        lr=common_args['learning_rate'],
        num_workers=common_args['num_workers'],
        eval_every=5,
        Cs=[1, 2, 0]
    )
    naive_usage(
        des="eval5/",
        dataset_name="voc",
        crit="RM",
        py_file="multi_label_batch_learning.py",
        seed_list=[156, 0],
        epoch=common_args["num_epochs"],
        batch_size=common_args["batch_size"],
        opt="sgd",
        cu=common_args['cuda'],
        lr=common_args['learning_rate'],
        num_workers=common_args['num_workers'],
        eval_every=5,
        Cs=[1, 2, 0]
    )