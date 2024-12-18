import os
from ray import tune
from networks_test.other.my_model_1.my_model_1 import MTAD_GAT
import json
from ray import train
from my_utils import *

# 全局变量，存储加载的数据
data = None


# 加载数据并将其存储在全局变量中
def load_data():
    global data
    if data is None:
        # 你的数据加载逻辑
        params, train_loader, val_loader, test_loader, anomaly_label = my_load_data()
        data = (params, train_loader, val_loader, test_loader, anomaly_label)
    return data


def train_model(config):
    name = train.get_context().get_trial_name()

    # 设置当前工作路径为文件实际路径
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # 获取数据
    params, train_loader, val_loader, test_loader, anomaly_label = load_data()

    # # 定义模型
    # model = MTAD_GAT(
    #     window_size=params["window_size"], next_steps=params["next_steps"],
    #     pre_kernel_size=(3,9),
    #     pre_nb_filters=20,
    #     pre_dilations=(1,2,4,8),
    #
    #     ae_kernel_size=(3,9),
    #     ae_dilations=(1,2,4,8),
    #     ae_nb_filters=20,
    #     ae_filters_conv1d=20,
    #     lamb = config["lamb"],
    # )
    from my_model.tcn_pred_para.tcn_pred import TCN_PRED
    model = TCN_PRED(
                 dilations=config["dilations"],
                 nb_filters=config["nb_filters"],
                 kernel_size=config["kernel_size"],
                 nb_stacks=config["nb_stacks"],
    )

    # 训练评估
    best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "tcn_pre_para/"+name)
    train.report({"best_f1": best_f1})

    # 将字典写入文件
    json_dir = os.path.join(params["model_root"], 'config.json')
    with open(json_dir, 'w') as file:
        json.dump(config, file)


if __name__ == '__main__':
    from itertools import product

    small_kernels = [3, 5, 7, 9, 11]
    large_kernels = [11, 13, 15,17, 19,21]

    kernel_sizes = [(s, l) for s, l in product(small_kernels, large_kernels) if s < l]
    # 定义搜索空间
    config = {
        "kernel_size": tune.choice(kernel_sizes),
        "nb_filters": tune.choice([i for i in range(3,21)]),  # 搜索不同的滤波器数量 3到21
        "dilations": tune.choice([(1, 2, 4, 8), (1, 2, 4), (1, 2)]),
        "nb_stacks": tune.choice([1, 2, 3]),

        # "lamb": tune.choice([1,2,3,4]),
    }

    # 配置 Ray Tune，指定优化目标
    from ray.tune.search.hyperopt import HyperOptSearch

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    analysis = tune.run(
        train_model,  # 训练函数
        config=config,  # 配置搜索空间
        metric="best_f1",  # 优化的指标
        mode="max",  # 最大化最佳 F1 分数
        num_samples=200,  # 运行 10 个不同的试验
        resources_per_trial={"gpu": 1},
        name="best_f1_tune",  # 试验名称
        search_alg=HyperOptSearch(),
        max_concurrent_trials=1,
    )

    # 输出最好的结果
    print("Best config found: ", analysis.best_config)
    print("Best F1 score: ", analysis.best_result["best_f1"])
