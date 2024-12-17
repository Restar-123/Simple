import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import yaml
sys.path.append("../../../")
import logging
from common import data_preprocess
from common.dataloader import load_dataset, get_dataloaders
from common.utils import seed_everything,set_logger, print_to_json
from common.evaluation import Evaluator
import torch


def my_load_data2(params):
    # seed_everything(2024)

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        valid_ratio=params["valid_ratio"],
        dim=params["dim"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        train_nrows=params["train_nrows"],
        test_nrows=params["test_nrows"]
    )

    # preprocessing
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # sliding windows
    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    # train/test on each entity put here
    evaluator = Evaluator(**params["eval"])
    entity = params["entities"][0]
    windows = window_dict[entity]
    train_windows = windows["train_windows"]
    test_windows = windows["test_windows"]
    # val_windows = windows["valid_windows"]
    train_loader, val_loader, test_loader = get_dataloaders(
        train_windows,
        test_windows,
        valid_data= None,
        next_steps=params["next_steps"],
        batch_size=params["batch_size"],
        shuffle=params["shuffle"],
        num_workers=params["num_workers"],
    )
    anomaly_label = (windows["test_label"].sum(axis=1) > 0).astype(int)

    return params, train_loader, val_loader, test_loader, anomaly_label


def my_load_data():
    # seed_everything(2024)
    with open('config_output.yaml', 'r') as file:
        params = yaml.safe_load(file)

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        valid_ratio=params["valid_ratio"],
        dim=params["dim"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        train_nrows=params["train_nrows"],
        test_nrows=params["test_nrows"]
    )

    # preprocessing
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # sliding windows
    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    # train/test on each entity put here
    evaluator = Evaluator(**params["eval"])
    entity = params["entities"][0]
    windows = window_dict[entity]
    train_windows = windows["train_windows"]
    test_windows = windows["test_windows"]
    # val_windows = windows["valid_windows"]
    train_loader, val_loader, test_loader = get_dataloaders(
        train_windows,
        test_windows,
        valid_data= None,
        next_steps=params["next_steps"],
        batch_size=params["batch_size"],
        shuffle=params["shuffle"],
        num_workers=params["num_workers"],
    )
    anomaly_label = (windows["test_label"].sum(axis=1) > 0).astype(int)

    return params, train_loader, val_loader, test_loader, anomaly_label



def my_train(model,params,train_loader,val_loader,test_loader,anomaly_label,idx=0):
    set_logger(params,str(idx))
    logging.info(print_to_json(params))
    if (params["need_training"]):
        model.fit(train_loader = train_loader, val_loader = val_loader, epochs = params["nb_epoch"], lr = params["lr"])
        torch.save(model.state_dict(), params["model_root"] + '/model.pth')
    else:
        model.load_state_dict(torch.load(params["model_root"] + '/model.pth'))

    train_anomaly_score = model.predict_prob(train_loader)
    anomaly_score = model.predict_prob(test_loader)
    score_dict = { "anomaly_score": anomaly_score, "anomaly_label": anomaly_label, "train_anomaly_score": train_anomaly_score}

    evaluator = Evaluator(**params["eval"])
    best_f1 = evaluator.eval_exp(
        score_dict = score_dict,
        params=params,
    )
    return best_f1