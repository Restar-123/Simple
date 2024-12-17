from my_utils import *

if __name__ == '__main__':
    params, train_loader, val_loader, test_loader, anomaly_label = my_load_data()

    # # aamp
    # from my_model.aamp.aamp import AAMP
    # model = AAMP(device=0,next_steps=params["next_steps"],window_size=params["window_size"],a=0.064)
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp")

    # # aamp_no_pred
    # from my_model.aamp.aamp_no_pred import AAMP_NO_PRED
    # model = AAMP_NO_PRED(device=0, next_steps=params["next_steps"])
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_pred")

    # # aamp_no_memory
    # from my_model.aamp.aamp_no_memory import AAMP_NO_MEMORY
    # model = AAMP_NO_MEMORY(device=0, next_steps=params["next_steps"], window_size=params["window_size"], a=0.06)
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_no_memory")

    # # aamp_single
    # from my_model.aamp.aamp_single import AAMP_SINGLE
    # model = AAMP_SINGLE(device=0,next_steps=params["next_steps"],window_size=params["window_size"],a=0.064)
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp_single")

    # anomaly_transformer
    from compare_model.anomaly_transformer import AnomalyTransformer

    # model = AnomalyTransformer(
    #     lr=0.0001,
    #     num_epochs=100,
    #     k=3,
    #     win_size=params["window_size"],
    #     input_c=params["dim"],
    #     output_c=params["dim"],readme
    #     batch_size=params["batch_size"],
    #     model_save_path=params["model_root"],
    #     device=0,
    # )
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "anomaly_transformer")

    # # lstm-vae
    # from compare_model.lstm_vae import LSTM_VAE
    # model = LSTM_VAE()
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "lstm_vae")

    # # madgan
    # from compare_model.madgan import MADGAN
    # model = MADGAN()
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "madgan")


