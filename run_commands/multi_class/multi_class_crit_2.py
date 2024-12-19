from run_commands.multi_label.usage import naive_usage

if __name__ == '__main__':
    naive_usage(
        dataset_name="tinyimagenet",
        crit="BCE",
        py_file="multi_class_batch_learning.py",
        seed_list=[2222],
        epoch=65,
        batch_size=192,
        opt="sgd",
        cu=3,
        lr=0.01,
    )


