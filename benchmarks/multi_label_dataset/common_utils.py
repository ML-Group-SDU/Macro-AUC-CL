import torch

task_class_squence = [21,81,81]
def make_common_onehot(spec_onehot, task_label, task_class_squence=task_class_squence):
    data_num = spec_onehot.shape[0]
    total_classes = sum(task_class_squence)
    common_onehot = torch.zeros([data_num,total_classes]).to(spec_onehot.device)
    if task_label == 0:
        common_onehot[:,:task_class_squence[0]] = spec_onehot
    elif task_label == 1:
        common_onehot[:,task_class_squence[0]:task_class_squence[0]+task_class_squence[1]] = spec_onehot
    elif task_label == 2:
        common_onehot[:,task_class_squence[0]+task_class_squence[1]:] = spec_onehot
    else:
        print("More Tasks is not implemented ! ! !")
    return common_onehot