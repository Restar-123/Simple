from my_utils import *

if __name__ == '__main__':
    params, train_loader, val_loader, test_loader, anomaly_label = my_load_data()
    from my_model.aamp.aamp import AAMP
    model = AAMP(device=0,next_steps=params["next_steps"])
    best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp")