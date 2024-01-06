import time
import math
import copy
import torch
import numpy as np
from flcore.clients.clientcdp import clientCDP
from flcore.servers.serverbase import Server
from flcore.privacy_analysis.RDP.compute_rdp import compute_rdp
from flcore.privacy_analysis.RDP.rdp_convert_dp import compute_eps
from copy import deepcopy
from threading import Thread


class CDPFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCDP)
        self.rdp = 0.
        self.orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
        self.q = 1  # 本地数据采样率
        self.local_epochs = args.local_epochs
        self.delta = args.delta
        self.batch_size = args.batch_size
        self.model = args.model
        self.l2_norm = args.l2_norm
        self.noise_scale = args.noise_scale

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        # self.global_model.train()
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            self.receive_models()
            # 初始化参数变化量列表
            parameter_changes = []
            # 初始化参数
            theta_r = self.global_model.state_dict()  # θr，初始参数
            for w, client in zip(self.uploaded_weights, self.selected_clients):
                delta_k = client.train()
                # 保存参数变化量
                for key in delta_k:
                    delta_k[key] *= w
                parameter_changes.append(delta_k)

            # 计算参数的加权平均变化量
            average_change = {}
            for param_name in theta_r.keys():
                delta_sum = sum([delta[param_name] for delta in parameter_changes])
                average_change[param_name] = delta_sum

            # 计算噪声的标准差
            sigma = self.noise_scale * self.l2_norm / self.q
            # 添加高斯噪声
            noise = {param_name: torch.randn(theta_r[param_name].shape) * sigma for param_name in theta_r.keys()}

            # 更新参数 θr+1
            theta_r_plus_1 = {}
            for param_name in theta_r.keys():
                theta_r_plus_1[param_name] = theta_r[param_name] + average_change[param_name] + noise[param_name]

            # 将更新后的参数应用到模型中
            self.global_model.load_state_dict(theta_r_plus_1, strict=True)

            self.rdp += compute_rdp(self.q, sigma, self.local_epochs * math.floor(1 / self.q), self.orders)
            epsilon, best_alpha = compute_eps(self.orders, self.rdp, self.delta)
            print("Iteration: {:3.0f}".format(i) + " | epsilon: {:7.4f}".format(
                epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        # if self.num_new_clients > 0:
        #     self.eval_new_clients = True
        #     self.set_new_clients(clientAVG)
        #     print(f"\n-------------Fine tuning round-------------")
        #     print("\nEvaluate new clients")
        #     self.evaluate()
