import logging
import os
import torch
import torch.nn as nn
import numpy as np

from common.utils import set_device
from .dlutils import ConvLSTM

## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
    def __init__(self, feats, window_size, lr, model_root, device):
        super(MSCRED, self).__init__()
        self.name = "MSCRED"
        self.name = "TranAD"
        self.n_feats = feats
        self.n_window = window_size
        self.lr = lr
        self.device = set_device(device)
        self.encoder = nn.ModuleList(
            [
                ConvLSTM(1, 32, (3, 3), 1, True, True, False),
                ConvLSTM(32, 64, (3, 3), 1, True, True, False),
                ConvLSTM(64, 128, (3, 3), 1, True, True, False),
            ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1),
            nn.Sigmoid(),
        )
        self.to(self.device)

    def forward(self, g):
        batch_size = g.shape[0]
        ## Encode
        z = g.view(batch_size, 1, self.n_window, self.n_feats)
        for cell in self.encoder:
            _, z = cell(z.unsqueeze(1))
            z = z[0][0]
        ## Decode
        x = self.decoder(z)
        x = x.view(batch_size, self.n_window, self.n_feats)
        return x

    def fit(self, epochs, train_loader, val_loader, lr, training=True):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        for epoch in range(1, epochs + 1):
            mse_func = nn.MSELoss(reduction="none")
            if training:
                logging.info("Training epoch: {}".format(epoch))
                for _, d in enumerate(train_loader):
                    d = d.to(self.device)
                    x = self(d)
                    loss = torch.mean(mse_func(x, d))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                logging.info("Epoch: {} finished.".format(epoch))

    def predict_prob(self, test_iterator, window_labels=None):
        with torch.no_grad():
            self.eval()
            mse_func = nn.MSELoss(reduction="none")
            loss_steps = []
            for d in test_iterator:
                d = d.to(self.device)
                x = self(d)
                loss = mse_func(x, d).view(-1, self.n_window, self.n_feats)
                loss_steps.append(loss.detach().cpu().numpy())
            anomaly_score = np.concatenate(loss_steps).mean(axis=(2, 1))
            if window_labels is None:
                return anomaly_score
            else:
                anomaly_label = (np.sum(window_labels, axis=1) >= 1) + 0
                return anomaly_score, anomaly_label
