# Taken from https://github.com/d-ailin/GDN/

import math
from typing import Sequence, Tuple

import torch
from torch import nn as nn
from torch.nn import Linear, Parameter, functional as F
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from common.utils import set_device
import logging
import time
import numpy as np


class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)

        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


def build_fc_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    edges = torch.tensor(edges, dtype=torch.long)
    edge_index = edges.T.contiguous()

    return edge_index


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num: int, hidden_dims: Sequence[int] = (512,), output_steps: int = 1):
        super(OutLayer, self).__init__()
        self.output_steps = output_steps

        modules = []

        dims = [in_num] + list(hidden_dims) + [output_steps]

        for i, (in_dims, out_dims) in enumerate(zip(dims[:-1], dims[1:])):
            # last layer, output shape:1
            modules.append(nn.Linear(in_dims, out_dims))
            if i != len(hidden_dims) - 1:
                modules.append(nn.BatchNorm1d(out_dims))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                # Note that batch norm expects (B, D, T) inputs, but also keep in mind that the roles
                # of time and features are reversed in this model
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)


def fallback_knn_graph(embedding: torch.Tensor, k: int):
    node_num = embedding.shape[0]
    weights = embedding.view(node_num, -1)

    cos_ji_mat = torch.matmul(weights, weights.T)
    normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
    cos_ji_mat = cos_ji_mat / normed_mat

    topk_indices_ji = torch.topk(cos_ji_mat, k, dim=-1)[1]

    gated_i = torch.arange(0, node_num, device=embedding.device).T.unsqueeze(1).repeat(1, k).flatten().unsqueeze(0)
    gated_j = topk_indices_ji.flatten().unsqueeze(0)
    return torch.cat((gated_j, gated_i), dim=0)


class GDN(nn.Module):
    def __init__(self, data_dim: int, window_size: int, next_steps:int=1, device =0, dim=64, out_layer_hidden_dims: Sequence[int] = (64,), topk=15,
                 dropout_prob: float = 0.2):
        """
        (Comments by Tobias) Terminology is a bit confusing here, so I'll add some explanations.

        :param data_dim: This is the number of features in the dataset, i.e., D!
        :param window_size: This is the length of a TS window, i.e, T!
        :param dim: The dimensionality of the embedding.
        :param out_layer_hidden_dims: Hidden dimensions for fully connected output layer
        :param topk: Number of edges that should be kept in the graph construction.
        """
        super(GDN, self).__init__()

        self.edge_index_sets = [build_fc_edge_index(data_dim)]

        embed_dim = dim
        self.embedding = nn.Embedding(data_dim, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(self.edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(window_size, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim * edge_set_num, out_layer_hidden_dims, output_steps= next_steps)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(dropout_prob)

        self.init_params()

        self.device = set_device(device)
        self.to(self.device)

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = inputs
        edge_index_sets = self.edge_index_sets
        device = x.device

        x = x.transpose(1, 2)
        batch_num, node_num, all_feature = x.shape
        x = x.reshape(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num, device=device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)
            if weights_arr.device.type == 'cpu':
                # torch-geometric knn_graph does not support CPU, so we keep the original method as a fallback
                gated_edge_index = fallback_knn_graph(weights_arr, self.topk)
            else:
                gated_edge_index = knn_graph(weights_arr, self.topk, cosine=True)

            self.learned_graph = gated_edge_index[0].view(-1, self.topk)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings)

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))

        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)

        # Outputs should be (B, 1, D)
        out = out.view(batch_num, 1, node_num)
        return out

    def fit(self, train_loader, val_loader, epochs, lr, criterion=nn.MSELoss()):
        fit_prediction(self, train_loader, epochs, lr, criterion)

    def predict_prob(self, dataloader, window_labels=None):
        self.eval()
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        with torch.no_grad():
            for input,label in dataloader:
                input = input.to(self.device)
                label = label.to(self.device)
                pred = self(input)
                loss = mse_func(label, pred)
                loss_steps.append(loss.detach().cpu().numpy())
        anomaly_scores = np.concatenate(loss_steps).mean(axis=(2, 1))
        if window_labels is not None:
            anomaly_label = (window_labels.sum(axis=1) > 0).astype(int)
            return anomaly_scores, anomaly_label
        else:
            return anomaly_scores

def fit_prediction(model, dataloader, epochs, lr, criterion=nn.MSELoss()):
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-5)
    train_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        for input,label in dataloader:
            input = input.to(model.device)
            label = label.to(model.device)

            pred = model(input)
            # 反向传播和优化
            optimizer.zero_grad()
            loss = criterion(pred, label)

            loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        s = (
            f"[Epoch {epoch + 1}] "
            f"loss = {epoch_loss / len(dataloader):.5f}, "
        )
        s += f" [{epoch_time:.2f}s]"
        logging.info(s)
        print(s)

    train_time = int(time.time() - train_start)
    logging.info(f"-- Training done in {train_time}s")