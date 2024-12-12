import sys
import numpy as np
import json
import logging
import h5py
import random
import os
import torch
import tensorflow as tf
import glob
import yaml
from collections import OrderedDict
from datetime import datetime

# add this line to avoid weird characters in yaml files
yaml.Dumper.ignore_aliases = lambda *args : True

DEFAULT_RANDOM_SEED = 2022

def set_cur_dir():
    # 获取脚本所在的目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取上两级目录路径
    upper_two_levels = os.path.abspath(os.path.join(script_dir, "../.."))
    # 设置当前工作目录为上两级目录
    os.chdir(upper_two_levels)
    sys.path.append("../../../")

def cal_path(params,model_root):
    model_type = params["model_type"]
    model_id = params["model_id"]
    dataset_id = params["dataset_id"]
    month = str(datetime.now().month)
    day = str(datetime.now().day)
    model_pth = os.path.join(model_root,model_type,model_id, dataset_id, month, day)
    os.makedirs(model_pth, exist_ok=True)
    return model_pth

def load_config(config_dir, experiment_id, model_type, eval_id=""):
    params = dict()
    params["model_type"] =model_type
    idx_underline = experiment_id.rfind("_")
    model_id = experiment_id[:idx_underline]

    model_config_dir = os.path.join(config_dir, "model_config.yaml")
    if not os.path.exists(model_config_dir):
        model_config_dir = config_dir + "model_config/" + model_type +'/'+ model_id  + '.yaml'
    if not os.path.exists(model_config_dir):
        raise RuntimeError("model_config_dir is not valid!")

    found_params = dict()
    with open(model_config_dir, "r") as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
        if "Base" in config_dict:
            found_params["Base"] = config_dict["Base"]
        if experiment_id in config_dict:
            found_params[experiment_id] = config_dict[experiment_id]
    if experiment_id not in found_params:
        raise ValueError("exp_id={} not found in config".format(experiment_id))
    # Update base settings first so that values can be overrided when conflict
    # with experiment_id settings
    params.update(found_params.get("Base", {}))
    params.update(found_params.get(experiment_id))
    params["exp_id"] = experiment_id
    dataset_params = load_dataset_config(config_dir, params["dataset_id"])
    params.update(dataset_params)
    if "eval" not in params:
        eval_params = load_eval_config(config_dir, params.get(eval_id, "Base"))
        params["eval"] = eval_params
    return params


def load_dataset_config(config_dir, dataset_id):
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/"+dataset_id+'.yaml'))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            try:
                config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
                if "Base" in config_dict:
                    dataset_config = config_dict["Base"]
                else:
                    dataset_config = {}
                if dataset_id in config_dict:
                    dataset_config.update(config_dict[dataset_id])
                    return dataset_config
            except TypeError:
                pass
    raise RuntimeError("dataset_id={} is not found in config.".format(dataset_id))


def load_eval_config(config_dir, eval_id="Base"):
    eval_configs = glob.glob(os.path.join(config_dir, "eval_config.yaml"))
    if not eval_configs:
        eval_configs = glob.glob(os.path.join(config_dir, "eval_config/*.yaml"))
    for config in eval_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            eval_config = config_dict["Base"]
            if eval_id in config_dict:
                eval_config.update(config_dict[eval_id])
                return eval_config
    raise RuntimeError("eval_id={} is not found in config.".format(eval_id))


def set_device(gpu=-1):
    import torch

    if gpu != -1 and torch.cuda.is_available():
        device = torch.device(
            "cuda:" + str(0)
        )  # already set env var in set logger function.
    else:
        device = torch.device("cpu")
    return device


def pprint(d, indent=0):
    d = sorted([(k, v) for k, v in d.items()], key=lambda x: x[0])
    for key, value in d:
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            pprint(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(round(value, 4)))


def load_hdf5(infile):
    logging.info("Loading hdf5 from {}".format(infile))
    with h5py.File(infile, "r") as f:
        return {key: f[key][:] for key in list(f.keys())}


def save_hdf5(outfile, arr_dict):
    logging.info("Saving hdf5 to {}".format(outfile))
    with h5py.File(outfile, "w") as f:
        for key in arr_dict.keys():
            f.create_dataset(key, data=arr_dict[key])


def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)


def load_json(infile):
    with open(infile, "r") as fr:
        return json.load(fr)


def update_from_nni_params(params, nni_params):
    if nni_params:
        params.update(nni_params)
    return params


def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def seed_tf(seed=DEFAULT_RANDOM_SEED):
    try:
        tf.random.set_seed(seed)
    except:
        tf.set_random_seed(seed)


def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_tf(seed)
    seed_torch(seed)


def set_logger(params, log, log_file=None):
    if log_file is None:
        log_dir = os.path.join(
            params["model_root"], # 根文件 ./details
            params["model_type"], # model 类型
            params["model_id"],  # model名字
            log,
        )
        log_file = os.path.join(log_dir, "execution.log")
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    params["model_root"] = log_dir
    params["uptime"] = datetime.now().strftime("%Y%m%d-%H%M%S")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # logging全局配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%m-%d %H:%M:%S",  # 不包含年份的时间格式
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )

    if params.get("device", -1) != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(params["device"])
        logging.info("Using device: cuda: {}".format(params["device"]))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logging.info("Using device: cpu.")
