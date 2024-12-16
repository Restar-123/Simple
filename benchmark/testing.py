from my_utils import *

if __name__ == '__main__':
    params, train_loader, val_loader, test_loader, anomaly_label = my_load_data()

    # aamp
    from my_model.aamp.aamp import AAMP
    model = AAMP(device=0,next_steps=params["next_steps"],window_size=params["window_size"],a=0.2)
    best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp")

    # # aamp_no_pred
    # from my_model.aamp.aamp_no_pred import AAMP_NO_PRED
    # model = AAMP_NO_PRED(device=0, next_steps=params["next_steps"])
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred")

    # # aamp_no_pred_memory
    # from my_model.aamp.aamp_no_pred_memory import AAMP_NO_PRED_MEMORY
    # model = AAMP_NO_PRED_MEMORY(device=0, next_steps=params["next_steps"])
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred_memory")

    # # aamp_no_pred_lsk
    #     # from my_model.aamp.aamp_no_pred_lsk import AAMP_NO_PRED_LSK
    #     # model = AAMP_NO_PRED_LSK(device=0, next_steps=params["next_steps"])
    #     # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred_lsk")

    # # aamp_no_pred_lsk_memory
    # from my_model.aamp.aamp_no_pred_lsk_memory import AAMP_NO_PRED_LSK_MEMORY
    # model = AAMP_NO_PRED_LSK_MEMORY(device=0, next_steps=params["next_steps"])
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred_lsk_memory")




