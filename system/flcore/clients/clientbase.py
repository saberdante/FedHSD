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

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)  # 设置随机种子
        self.model = copy.deepcopy(args.model)  # 深拷贝模型
        self.algorithm = args.algorithm  # 算法类型
        self.dataset = args.dataset  # 数据集
        self.device = args.device  # 设备（CPU或GPU）
        self.id = id  # 客户端ID
        self.save_folder_name = args.save_folder_name  # 保存文件夹名称

        self.num_classes = args.num_classes  # 类别数量
        self.train_samples = train_samples  # 训练样本
        self.test_samples = test_samples  # 测试样本
        self.batch_size = args.batch_size  # 批量大小
        self.learning_rate = args.local_learning_rate  # 学习率
        self.local_epochs = args.local_epochs  # 本地训练轮数

        # 检查是否有BatchNorm层
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']  # 训练速度
        self.send_slow = kwargs['send_slow']  # 发送速度
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}  # 训练时间成本
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}  # 发送时间成本

        self.loss = nn.CrossEntropyLoss()  # 损失函数
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)  
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma  
        )
        self.learning_rate_decay = args.learning_rate_decay  

        self.historical_max_acc = [0] * args.num_classes  


        self.initial_acc = 0.0

        self.last_round_acc = 0.0
        self.knowledge_drift = 0.0


        self.test_correct_per_class_history = []
        self.test_total_per_class_history = []
        
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True) 
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)  

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)  
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)  
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()  

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone() 
            
    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()  

    def test_metrics(self):
        testloaderfull = self.load_test_data()  
        self.model.eval()  

        test_acc = 0  
        test_num = 0 
        y_prob = []  
        y_true = []  
        
        with torch.no_grad():  
            for x, y, idx in testloaderfull:  
                if type(x) == type([]): 
                    x[0] = x[0].to(self.device)  
                else:
                    x = x.to(self.device)  
                y = y.to(self.device)  
                output = self.model(x) 


                if isinstance(output, (list, tuple)) and len(output) == 2:
                    output_for_acc = output[0]
                else:
                    output_for_acc = output

                test_acc += (torch.sum(torch.argmax(output_for_acc, dim=1) == y)).item()  
                test_num += y.shape[0] 
                y_prob.append(output_for_acc.detach().cpu().numpy()) 
                nc = self.num_classes  
                if self.num_classes == 2:
                    nc += 1  
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc)) 
                if self.num_classes == 2:
                    lb = lb[:, :2] 
                y_true.append(lb)  

        y_prob = np.concatenate(y_prob, axis=0)  
        y_true = np.concatenate(y_true, axis=0)  

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc  

    def train_metrics(self):
        trainloader = self.load_train_data()  
        self.model.eval()  

        train_num = 0
        losses = 0
        with torch.no_grad():  
            for x, y, idx in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                if isinstance(output, (list, tuple)) and len(output) == 2:
                    output_for_loss = output[0]
                else:
                    output_for_loss = output
                loss = self.loss(output_for_loss, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num  

    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))  

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt")) 

    def get_current_acc_per_class(self):

        testloader = self.load_test_data()
        self.model.eval()

        correct_per_class = [0] * self.num_classes
        total_per_class = [0] * self.num_classes

        with torch.no_grad():
            for x, y, idx in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                if isinstance(output, (list, tuple)) and len(output) == 2:
                    output_for_acc = output[0]
                else:
                    output_for_acc = output
                preds = torch.argmax(output_for_acc, dim=1)

                for i in range(self.num_classes):
                    correct_per_class[i] += ((preds == i) & (y == i)).sum().item()
                    total_per_class[i] += (y == i).sum().item()


        self.test_correct_per_class_history.append(list(correct_per_class))
        self.test_total_per_class_history.append(list(total_per_class))


        total_samples = sum(total_per_class)
        weights = [
            (total / total_samples) if total_samples > 0 else 0
            for total in total_per_class
        ]


        current_acc_per_class1 = [
            (correct / total if total > 0 else 0) 
            for correct, total in zip(correct_per_class, total_per_class)
        ]
        
        current_acc_per_class = [
            (correct / total if total > 0 else 0) * weight
            for correct, total, weight in zip(correct_per_class, total_per_class, weights)
        ]
        

        self.historical_max_acc = [
            max(historical, current)
            for historical, current in zip(self.historical_max_acc, current_acc_per_class)
        ]

        return current_acc_per_class, current_acc_per_class1

    def record_initial_acc(self):

        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, idx in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                if isinstance(output, (list, tuple)) and len(output) == 2:
                    output_for_acc = output[0]
                else:
                    output_for_acc = output
                preds = torch.argmax(output_for_acc, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        self.initial_acc = correct / total if total > 0 else 0

    def record_last_round_acc(self):

        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, idx in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                if isinstance(output, (list, tuple)) and len(output) == 2:
                    output_for_acc = output[0]
                else:
                    output_for_acc = output
                preds = torch.argmax(output_for_acc, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        self.last_round_acc = correct / total if total > 0 else 0

    def get_knowledge_drift(self):

        if self.last_round_acc is not None and self.initial_acc is not None:
            return self.last_round_acc - self.initial_acc
        else:
            return 0.0

    def save_test_per_class_history(self, save_dir=None):

        import pandas as pd
        import os
        if save_dir is None:
            save_dir = os.path.join("../acc_result", f"{self.dataset}_{self.algorithm}_client_per_class")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存correct_per_class
        df_correct = pd.DataFrame(self.test_correct_per_class_history)
        df_correct.index.name = 'Round'
        df_correct.columns = [f'Class_{i}' for i in range(self.num_classes)]
        df_correct.to_csv(os.path.join(save_dir, f"client_{self.id}_correct_per_class.csv"))
        # 保存total_per_class
        df_total = pd.DataFrame(self.test_total_per_class_history)
        df_total.index.name = 'Round'
        df_total.columns = [f'Class_{i}' for i in range(self.num_classes)]
        df_total.to_csv(os.path.join(save_dir, f"client_{self.id}_total_per_class.csv"))

