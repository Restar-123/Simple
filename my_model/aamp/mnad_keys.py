import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
from torch.nn import functional as F

class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather

        # Initialize keys as learnable parameters
        self.keys = nn.Parameter(torch.randn(memory_size, key_dim))  # 初始化 keys 为可训练的参数
        self.keys.requires_grad = True  # 确保 keys 会在训练中更新

    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem, torch.t(self.keys))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)

        return self.keys[max_idx]

    def random_pick_memory(self, mem, max_indices):
        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)

        return torch.tensor(output)

    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        m, d = mem.size()
        if train:
            query_update = torch.zeros((m, d)).cuda()
            random_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                else:
                    query_update[i] = 0

            return query_update

        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                else:
                    query_update[i] = 0

            return query_update

    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X h X w X m
        score = score.view(bs * h * w, m)  # (b X h X w) X m

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)

        return score_query, score_memory

    def forward(self, query, train=True):
        keys = self.keys
        batch_size, dims, h, w = query.size()  # b X d X h X w
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d

        # train
        if train:
            # gathering loss
            gathering_loss = self.gather_loss(query, keys, train)
            # spreading_loss
            spreading_loss = self.spread_loss(query, keys, train)
            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            # update
            updated_memory = self.update(query, keys, train)

            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss

        # test
        else:
            # gathering loss
            gathering_loss = self.gather_loss(query, keys, train)

            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)

            # update
            updated_memory = self.keys  # use the current keys as memory during inference

            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss

    def update(self, query, keys, train):
        batch_size, h, w, dims = query.size()  # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

        if train:
            # top-1 queries (of each memory) update (weighted sum) & random pick
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + self.keys, dim=1)

        else:
            # only weighted sum update when test
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + self.keys, dim=1)

        return updated_memory.detach()

    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()  # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')

        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return pointwise_loss

    def gather_loss(self, query, keys, train):
        batch_size, h, w, dims = query.size()  # b X h X w X d

        loss_mse = torch.nn.MSELoss(reduction='none')  # 使用 'none' 来逐样本计算损失

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        # 计算逐样本的损失
        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        # 将损失转换为 (batch_size, 1) 形状
        gathering_loss = gathering_loss.view(batch_size, -1).mean(dim=1, keepdim=True)  # 每个 batch 求均值

        return gathering_loss

    def spread_loss(self, query, keys, train):
        batch_size, h, w, dims = query.size()  # b X h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0, reduction='none')  # 使用 'none' 来逐样本计算损失

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, 1]]

        # 计算逐样本的损失
        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())

        # 将损失转换为 (batch_size, 1) 形状
        spreading_loss = spreading_loss.view(batch_size, -1).mean(dim=1, keepdim=True)  # 每个 batch 求均值

        return spreading_loss

    def read(self, query, updated_memory):
        batch_size, h, w, dims = query.size()  # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)  # (b X h X w) X d
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2)

        return updated_query, softmax_score_query, softmax_score_memory

if __name__ == '__main__':
    # model = Memory(10, 512, 512, 0.1, 0.1)
    # m_items =torch.rand(10, 512)
    # input = torch.rand(64,30,30,512)
    # ouput = model(input,m_items)

    # 创建 Memory 模块
    memory_module = Memory(10, 8, 8, temp_update=0.1, temp_gather=0.1).cuda()

    # 创建查询数据和记忆数据
    query = torch.randn(12, 8, 1, 32).cuda()  # 查询数据

    # 运行 Memory 模块
    updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = memory_module(query, train=True)


    # 输出结果
    print(query.shape)
    print(updated_query.shape)

    print(gathering_loss.shape)
    print(spreading_loss.shape)