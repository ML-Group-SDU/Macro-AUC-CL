import os
import yaml


def read_yaml(yaml_name):
    project_path, _ = get_project_path_and_name()
    with open(project_path + f"/configs/{yaml_name}", 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config

def get_project_path_and_name():
    current_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(current_file_path))
    project_name = os.path.basename(project_path)
    return project_path+"/", project_name


def get_log_path():
    project_path, project_name = get_project_path_and_name()
    config = read_yaml('paths.yaml')
    return os.path.expanduser("~") + config["save_dir"] + project_name + "/logs/"


def get_python_exec_path(py_exec_file):
    return os.path.expanduser("~") + py_exec_file

if __name__ == '__main__':
    print(get_log_path())
