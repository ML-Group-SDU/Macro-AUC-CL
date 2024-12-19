import argparse
import os

def base_opt_parser():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--opt', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # sgd
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum for SGD.(None means the default in optm)")
    parser.add_argument('--nesterov', action="store_true")
    # adam
    parser.add_argument('--betas', type=float, default=None, nargs='+',
                        help="Betas for AdamW Optimizer.(None means the default in optm)")
    parser.add_argument('--eps', type=float, default=None,
                        help="Epsilon for AdamW Optimizer.(None means the default in optm)")
    return parser


def sam_opt_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rho', type=float, default=0.05, help="Perturbation intensity of SAM type optims.")
    parser.add_argument('--sparsity', type=float, default=0.2,
                        help="The proportion of parameters that do not calculate perturbation.")
    parser.add_argument('--update_freq', type=int, default=5, help="Update frequency (epoch) of sparse SAM.")

    parser.add_argument('--num_samples', type=int, default=1024,
                        help="Number of samples to compute fisher information. Only for `ssam-f`.")
    parser.add_argument('--drop_rate', type=float, default=0.5, help="Death Rate in `ssam-d`. Only for `ssam-d`.")
    parser.add_argument('--drop_strategy', type=str, default='gradient', help="Strategy of Death. Only for `ssam-d`.")
    parser.add_argument('--growth_strategy', type=str, default='random', help="Only for `ssam-d`.")
    return parser

def my_opt_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=str,default="5", help="Select zero-indexed cuda device. -1 to use CPU.", )
    parser.add_argument("--do_initial", type=bool, default=False, )
    parser.add_argument("--eval_on_random", type=bool, default=False, )
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--txtlog_name', type=str, default='default.txt')
    parser.add_argument('--csvlog_dir', type=str, default='debug')
    parser.add_argument('--dataset_name', type=str, default='voc')
    parser.add_argument('--task_num', type=int, default=5)
    parser.add_argument("--crit", type=str, default='RM')
    parser.add_argument("--measure", type=str, default='macro-auc')
    parser.add_argument("--use_aug", type=str, default='no')
    parser.add_argument('--mem_size', type=int, default=2000)
    parser.add_argument('--imbalance',type=str,default="yes")
    parser.add_argument('--model',type=str,default='res34')
    parser.add_argument('--C',type=float,default=0.6)
    parser.add_argument('--nth_power', type=float, default=1/4)
    parser.add_argument('--num_workers',type=int,default=1),
    parser.add_argument('--with_wrs', type=str,default="yes")
    parser.add_argument('--memory_update', type=str, default="wrs")
    parser.add_argument('--scale_ratio', type=float, default=1.0)
    parser.add_argument('--eval_every', type=int, default=0)
    return parser

def ddp_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--nproc_per_node', type=int,default=1)
    parser.add_argument('--nnode', type=int,default=2)
    parser.add_argument('--node_rank', type=int,default=0)
    return parser

def get_args(out_parsers=None):
    all_parser_funcs = [base_opt_parser, sam_opt_parser, my_opt_parser,ddp_parser]

    all_parsers = [parser_func() for parser_func in all_parser_funcs]
    if out_parsers:
        all_parsers.append(out_parsers)
    final_parser = argparse.ArgumentParser(parents=all_parsers)
    args = final_parser.parse_args()

    if not os.path.exists(args.csvlog_dir):
        os.makedirs(args.csvlog_dir)
    f = open(args.csvlog_dir + "/" + args.txtlog_name, 'a+')
    print(
        str(args),
        file=f,
        flush=True,
    )
    f.close()
    return args

if __name__ == '__main__':
    print(get_args())