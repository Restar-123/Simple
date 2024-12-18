import logging
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from common.evaluation import adjust_pred


def compute_binary_metrics(anomaly_pred, anomaly_label, adjustment=False):
    if not adjustment:
        eval_anomaly_pred = anomaly_pred
        metrics = {
            "ac": accuracy_score(anomaly_label, eval_anomaly_pred),
            "pc": precision_score(anomaly_label, eval_anomaly_pred),
            "rc": recall_score(anomaly_label, eval_anomaly_pred),
            "f1": f1_score(anomaly_label, eval_anomaly_pred),
        }
    else:
        eval_anomaly_pred = adjust_pred(anomaly_pred, anomaly_label)
        metrics = {
            "ac_adjusted": accuracy_score(anomaly_label, eval_anomaly_pred),
            "pc_adjusted": precision_score(anomaly_label, eval_anomaly_pred),
            "rc_adjusted": recall_score(anomaly_label, eval_anomaly_pred),
            "f1_adjusted": f1_score(anomaly_label, eval_anomaly_pred),
        }
    return metrics


def compute_delay(anomaly_pred, anomaly_label):
    def onehot2interval(arr):
        result = []
        record = False
        for idx, item in enumerate(arr):
            if item == 1 and not record:
                start = idx
                record = True
            if item == 0 and record:
                end = idx  # not include the end point, like [a,b)
                record = False
                result.append((start, end))
        return result

    count = 0
    total_delay = 0
    pred = np.array(anomaly_pred)
    label = np.array(anomaly_label)
    for start, end in onehot2interval(label):
        pred_interval = pred[start:end]
        if pred_interval.sum() > 0:
            delay = np.where(pred_interval == 1)[0][0]
            delay = delay / len(pred_interval)  # normalized by the interval
            total_delay += delay
            count += 1
    avg_delay = total_delay / (1e-6 + count)
    return avg_delay
