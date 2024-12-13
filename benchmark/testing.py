from my_utils import *

if __name__ == '__main__':
    params, train_loader, val_loader, test_loader, anomaly_label = my_load_data()

    from my_model.aamp.aamp import AAMP
    model = AAMP(device=0,next_steps=params["next_steps"])

    best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "aamp")


    # model = MTAD_GAT(window_size=params["window_size"], next_steps=params["next_steps"],
    #                  pre_kernel_size=(3, 9),
    #                  pre_nb_filters=20,
    #                  pre_dilations=(1, 2, 4, 8),
    #
    #                  ae_kernel_size=(3, 9),
    #                  ae_dilations=(1, 2, 4, 8),
    #                  ae_nb_filters=20,
    #                  ae_filters_conv1d=20,
    #                  )

    # from my_model.tcn_ae_para.tcn_ae import TCN_AE
    # model = TCN_AE(use_skip_connections=True,device=0)
    #
    # from my_model.tcn_pred_para.tcn_pred import TCN_PRED

    # model = TCN_PRED(
    #     next_steps=params["next_steps"],
    #     dilations=(1,2,4),
    #     nb_filters=11,
    #     kernel_size=(21,21),
    #     nb_stacks=1,
    # )

    # from my_model.tcn_pred_single.tcn_pred import TCN_PRED
    # model = TCN_PRED()

    # from my_model.dual_flow_tcn_single.dual_flow_tcn_single import Dual_Flow_TCN
    # model = Dual_Flow_TCN()
    #

    # from my_model.dual_flow_tcn_para.dual_flow_tcn_para import Dual_Flow_TCN
    # model = Dual_Flow_TCN(
    #     next_steps=params["next_steps"],
    #     pre_kernel_size=(3, 9),
    #     pre_nb_filters=20,
    #     pre_dilations=(1, 2, 4, 8),
    # )

    # best_f1 = my_train(model, params, train_loader, val_loader, test_loader, anomaly_label, "tcn_ae_para")