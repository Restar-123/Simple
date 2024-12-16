from my_utils import *

if __name__ == '__main__':
    params, train_loader, val_loader, test_loader, anomaly_label = my_load_data()

    # from networks.reconstruction.usad import UsadModel
    # model = UsadModel(w_size=params["window_size"] * params["dim"],
    #         z_size=params["window_size"] * 64,
    #         device= 0 ,
    #                   )
    #
    # from networks.reconstruction.anomaly_transformer import AnomalyTransformer
    #
    # model = AnomalyTransformer(
    #     lr=0.0001,
    #     num_epochs=20,
    #     k=3,
    #     win_size=params["window_size"],
    #     input_c=params["dim"],
    #     output_c=params["dim"],
    #     batch_size=params["batch_size"],
    #     model_save_path=params["model_root"],
    #     device=0,
    # )
    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "anomaly_transformer")

    from networks.reconstruction.dcdetector import DCdetector
    model = DCdetector(win_size=params["window_size"],
            enc_in=params["dim"],
            c_out=params["dim"],
            channel=params["dim"],)
    best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "mtad/dcdetector")