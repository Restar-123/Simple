import torch
import torch.nn as nn
from common.utils import set_device

import logging
import time
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class UsadModel(nn.Module):
    def __init__(self, w_size, z_size, device):
        super().__init__()
        self.w_size = w_size
        self.z_size = z_size
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
        self.device = set_device(device)
        self.to(self.device)

    def forward(self, batch):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1,w2,w3
    def training_step(self,batch,n):
        batch = batch.view(batch.shape[0],-1)
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1 / n * torch.mean((batch - w1) ** 2) + (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        loss2 = 1 / n * torch.mean((batch - w2) ** 2) - (1 - 1 / n) * torch.mean((batch - w3) ** 2)
        return loss1,loss2


    def fit(self, train_loader, val_loader, epochs, lr, ):
        fit_usad(self,train_loader,epochs,lr)

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        alpha = 0.5
        beta = 1-alpha
        with torch.no_grad():
            for input in dataloader:
                input = input.view(input.shape[0],-1)
                input = input.to(self.device)
                w1,w2,w3 = self(input)
                loss = alpha * mse_func(input, w1) + beta * mse_func(input,w2)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps).mean(axis=(1))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores


def fit_usad(model, train_loader, epochs, lr, criterion=nn.MSELoss()):
    opt_func=torch.optim.Adam
    optimizer1 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder1.parameters())
    )
    optimizer2 = opt_func(
        list(model.encoder.parameters()) + list(model.decoder2.parameters())
    )
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        model.train()
        for input in train_loader:
            input = input.to(model.device)
            loss1,loss2 = model.training_step(input,epoch+1)

            # 反向传播和优化
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss1, loss2 = model.training_step(input, epoch + 1)
            # 反向传播和优化
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # 累计损失
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

        epoch_time = time.time() - epoch_start
        s = (
            f"[Epoch {epoch + 1}] "
            f"loss_1 = {epoch_loss1 / len(train_loader):.5f}, "
            f"loss_2 = {epoch_loss2 / len(train_loader):.5f}, "
        )
        s += f" [{epoch_time:.2f}s]"
        logging.info(s)

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")
