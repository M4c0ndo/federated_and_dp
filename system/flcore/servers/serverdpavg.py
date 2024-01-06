import time
import math
from flcore.clients.clientdp import clientDP
from flcore.servers.serverbase import Server
from flcore.privacy_analysis.RDP.compute_rdp import compute_rdp
from flcore.privacy_analysis.RDP.rdp_convert_dp import compute_eps
from threading import Thread


class DPFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDP)
        self.rdp = 0.
        self.orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
        self.q = 1  # 本地数据采样率
        self.local_epochs = args.local_epochs
        self.sigma = args.sigma
        self.delta = args.delta

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.rdp += compute_rdp(self.q, self.sigma, self.local_epochs * math.floor(1/self.q), self.orders)
            epsilon, best_alpha = compute_eps(self.orders, self.rdp, self.delta)
            print("Iteration: {:3.0f}".format(i) + " | epsilon: {:7.4f}".format(
                epsilon) + " | best_alpha: {:7.4f}".format(best_alpha))

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

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
