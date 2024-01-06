import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from copy import deepcopy


class clientDP(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

    def train(self):
        # global privacy_engine, model_origin
        trainloader = self.load_train_data()
        batch_size = self.batch_size
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        noise_scale = self.l2_norm * self.sigma

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
                self.optimizer.zero_grad()
                temp_par = {}
                for p in self.model.named_parameters():
                    temp_par[p[0]] = torch.zeros_like(p[1])
                for p in self.model.named_parameters():
                    p[1].grad = torch.zeros_like(p[1])
                loss = self.loss(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.l2_norm, norm_type=2)
                for p in self.model.named_parameters():
                    temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
                for p in self.model.named_parameters():
                    noise = torch.normal(mean=0, std=noise_scale, size=temp_par[p[0]].size()).to(self.device)
                    p[1].grad = deepcopy(temp_par[p[0]]) + noise
                    p[1].grad = p[1].grad / batch_size
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
