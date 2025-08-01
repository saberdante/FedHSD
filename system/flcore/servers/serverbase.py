# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import pandas as pd  # 添加pandas导入
from utils.data_utils import read_client_data
from utils.dlg import DLG
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Server(object):
    def __init__(self, args, times):

        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new


        self.client_acc_history = {i: [] for i in range(self.num_clients)}


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)


    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients


    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)


    def select_clients(self):


        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            
            self.current_num_join_clients = self.num_join_clients
        
        
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def receive_models(self):
        assert (len(self.selected_clients) > 0) 

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))  

        self.uploaded_ids = []  #
        self.uploaded_weights = []  # 
        self.uploaded_models = []  # 
        tot_samples = 0  # 
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']  
            except ZeroDivisionError:
                client_time_cost = 0  
            if client_time_cost <= self.time_threthold: 
                tot_samples += client.train_samples 
                self.uploaded_ids.append(client.id)  
                self.uploaded_weights.append(client.train_samples)  
                self.uploaded_models.append(client.model)  
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples  

    
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)  

        self.global_model = copy.deepcopy(self.uploaded_models[0])  
        for param in self.global_model.parameters():
            param.data.zero_()  
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models): 
            self.add_parameters(w, client_model)  


    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w


    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)


    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)


    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)


    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def test_metrics(self):


        if self.eval_new_clients and self.num_new_clients > 0:

            self.fine_tuning_new_clients()

            return self.test_metrics_new_clients()
        
        num_samples = [] 
        tot_correct = []  
        tot_auc = []  
        for c in self.clients:

            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)  
            tot_auc.append(auc*ns)  
            num_samples.append(ns)  

        ids = [c.id for c in self.clients] 

        return ids, num_samples, tot_correct, tot_auc  


    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses


    def evaluate(self, acc=None, loss=None, decimal_places=4):


        stats = self.test_metrics()

        stats_train = self.train_metrics()


        selected_stats = self.test_metrics_selected_clients()


        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]


        selected_accs = [a / n for a, n in zip(selected_stats[2], selected_stats[1])]
        

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)


        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        client_accs = [round(acc, decimal_places) for acc in accs]
        print("Clients ID: ", stats[0])
        print("Clients Acc: ", client_accs)


        selected_client_accs = [round(acc, decimal_places) for acc in selected_accs]
        print("Selected Clients ID: ", selected_stats[0])
        print("Selected Clients Acc: ", selected_client_accs)


        print("\nKnowledge Drift Components:")
        selected_last_round_accs = {}
        selected_initial_accs = {}
        for client in self.selected_clients:
            print(f"Client {client.id}:")
            print(f"  - Last Round Acc: {client.last_round_acc:.4f}")
            print(f"  - Initial Acc: {client.initial_acc:.4f}")
            if client.id in [2, 3, 5, 7]:
                selected_last_round_accs[f"Client {client.id} Last Round Acc"] = client.last_round_acc
                selected_initial_accs[f"Client {client.id} Initial Acc"] = client.initial_acc


        forgetting_scores = []
        total_samples = sum(client.train_samples + client.test_samples for client in self.selected_clients)
        

        current_round = len(self.rs_test_acc) + 1


        client_forgetting_scores = {}
        for client in self.selected_clients:
            historical_max_acc = client.historical_max_acc
            current_acc_per_class, current_acc_per_class1 = client.get_current_acc_per_class()
            forgetting_score = sum((historical_max_acc[c] - current_acc_per_class[c]) for c in range(self.num_classes))
            

            for cid in range(self.num_clients):
                if cid == client.id:
                    self.client_acc_history[cid].append(current_acc_per_class1)
                elif len(self.client_acc_history[cid]) < current_round:

                    self.client_acc_history[cid].append([np.nan] * self.num_classes)

            if client.id in [2, 3, 5, 7]:
                key = f"Client {client.id} Forgetting"
                client_forgetting_scores[key] = forgetting_score
            

            weighted_forgetting = forgetting_score * ((client.train_samples + client.test_samples) / total_samples)
            forgetting_scores.append(weighted_forgetting)
        
        avg_forgetting = sum(forgetting_scores) if forgetting_scores else 0
        

        drift_scores = []
        for client in self.clients:
            drift = client.knowledge_drift 
            drift_scores.append(round(drift, decimal_places))
        avg_drift = np.mean(drift_scores) if drift_scores else 0
        print("Clients Knowledge Drift: ", drift_scores)
        print("Average Clients Knowledge Drift: {:.4f}".format(avg_drift))


        selected_client_drifts = {}
        for client in self.selected_clients:
            drift = client.knowledge_drift 
            if client.id in [2, 3, 5, 7]:
                selected_client_drifts[client.id] = drift
                print(f"Client {client.id} Forgetting: ", drift)


        current_round = len(self.rs_test_acc)
        if current_round == self.global_rounds:
            self.save_client_acc_history()



    def save_client_acc_history(self):

        folder_name = f"{self.dataset}_{self.algorithm}_client_acc_history"
        save_dir = os.path.join("../acc_result", folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for client_id in range(self.num_clients):
            history = self.client_acc_history[client_id]
            if not history:  
                continue
                

            rounds = range(1, len(history) + 1)
            df = pd.DataFrame(history, index=rounds)
            df.index.name = 'Round'
            df.columns = [f'Class_{i}' for i in range(self.num_classes)]
            

            base_filename = f"client_{client_id}_acc_history"
            # df.to_excel(os.path.join(save_dir, f"{base_filename}.xlsx"))
            df.to_csv(os.path.join(save_dir, f"{base_filename}.csv"))
            






    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))


    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y, idx) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')


    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)


    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y, idx) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
    
# ===== 2025-03-06 ===== #
    def test_metrics_selected_clients(self):

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.selected_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, tot_correct, tot_auc




    def avg_generalization_metrics(self):
        
        num_samples = []
        tot_correct = []
        tot_auc = []

        gen_acc = []

        loaders = []
        for c in self.clients:
            loaders.append(c.load_test_data())

        for client in self.clients:
            acc4c = []
            for loader in loaders:
                acc, num, _ = client.test_with_loader(loader)
                acc4c.append(acc/num)
            # print(acc4c)
            gen_acc.append(sum(acc4c)/len(acc4c))

        return sum(gen_acc)/len(gen_acc)
