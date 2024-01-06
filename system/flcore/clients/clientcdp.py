import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from copy import deepcopy
from utils.privacy import *


class clientCDP(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        parameter_changes = []  # 初始化保存参数变化量的列表
        prev_params = self.model.state_dict()  # 初始化prev_params为当前参数
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 保存参数的变化量
                current_params = self.model.state_dict()
                param_changes = {}
                for param_name in current_params.keys():
                    param_changes[param_name] = current_params[param_name] - prev_params[param_name]
                parameter_changes.append(param_changes)
                prev_params = current_params

                # 对当前参数进行裁剪操作
                # param_changes_tensor = torch.cat(list(param_changes.values())).flatten()
                # delta_norm = torch.norm(param_changes_tensor)
                norm_sum = 0.0
                for tensor in param_changes.values():
                    norm_sum += torch.norm(tensor) ** 2
                delta_norm = torch.sqrt(norm_sum)
                scaling_factor = min(1, self.l2_norm / delta_norm)
                for param_name, param in self.model.named_parameters():
                    param.data.copy_(prev_params[param_name] + scaling_factor * param_changes[param_name])


        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # 计算参数的差异
        parameter_diff = {}
        for param_name, param in self.model.named_parameters():
            parameter_diff[param_name] = param.data - prev_params[param_name]

        return parameter_diff
