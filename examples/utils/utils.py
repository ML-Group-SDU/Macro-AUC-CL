from os.path import expanduser
import random
import numpy as np
import torch
import os

root_path = expanduser("~") + "/avalanche/"
models_path = root_path + "model_files/"

tiny_class_order = [195, 144, 36, 11, 170, 153, 44, 72, 15, 168, 126, 151, 70, 130,
                         137, 65, 22, 92, 1, 42, 9, 8, 19, 115, 77, 123, 118, 182,
                         108, 160, 178, 93, 80, 18, 48, 2, 99, 122, 24, 152, 63, 179,
                         166, 75, 69, 154, 159, 119, 172, 14, 162, 31, 40, 135, 184, 94,
                         158, 25, 89, 68, 147, 155, 175, 98, 186, 60, 134, 67, 97, 197,
                         86, 56, 109, 34, 106, 96, 164, 28, 58, 143, 128, 90, 145, 102,
                         176, 38, 189, 190, 121, 57, 51, 55, 163, 138, 169, 188, 177, 104,
                         148, 61, 136, 114, 174, 111, 79, 7, 13, 87, 192, 101, 54, 180,
                         113, 161, 194, 193, 83, 142, 131, 129, 64, 171, 105, 146, 73, 112,
                         29, 37, 12, 82, 78, 150, 199, 81, 95, 45, 185, 141, 157, 39,
                         10, 125, 43, 107, 198, 76, 26, 120, 117, 173, 187, 140, 110, 181,
                         4, 33, 183, 62, 49, 17, 91, 50, 127, 6, 196, 132, 47, 53,
                         30, 156, 27, 5, 3, 165, 71, 116, 133, 88, 167, 0, 16, 23,
                         191, 35, 59, 32, 85, 46, 74, 21, 103, 52, 100, 149, 20, 124,
                         139, 66, 41, 84]

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def set_gpu_device(cuda_num):
    # os.environ["CUDA_VISIBLE_DEVICES"] = f'{cuda_num}'
    device = torch.device(
        f"cuda:{cuda_num}"
        if torch.cuda.is_available() and cuda_num >= 0
        else None
    )
    return device