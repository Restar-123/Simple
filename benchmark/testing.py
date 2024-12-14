from my_utils import *

if __name__ == '__main__':
    params, train_loader, val_loader, test_loader, anomaly_label = my_load_data()
    # from my_model.aamp.aamp2_g import AAMP
    # model = AAMP(device=0,next_steps=params["next_steps"])
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp2_no_g_s")

    # from my_model.aamp.no_memory import No_Memory
    # model = No_Memory(device = 0, )
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "no_memory")

    # from my_model.aamp.aamp_no_pred import AAMP
    # model = AAMP(device=0)
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred")

    # from my_model.aamp.aamp_no_pred_memory import AAMP
    # model = AAMP(device=0)
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred_memory")

    from my_model.aamp.aamp_no_pred_lsk import AAMP
    model = AAMP(device=0)
    best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred_lsk")