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
import math
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client  

class clientHSD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs) 
        self.prev_soft_labels = torch.zeros((train_samples, self.num_classes), device=self.device)  
        self.loss_mse = nn.MSELoss()
        self.beta = args.beta  
        self.lamda = args.lamda  
        self.sigma = args.sigma   
        self.softmax = torch.nn.Softmax(dim=1)
        self.global_features = None  

    def train(self):
        trainloader = self.load_train_data()  
        self.model.train()  
        
        start_time = time.time()  

        max_local_epochs = self.local_epochs  
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)  
        
        for param in self.model.base.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = False
        
        for i, (x, y, idx) in enumerate(trainloader):
            if type(x) == type([]):  
                x[0] = x[0].to(self.device) 
            else:
                x = x.to(self.device)
            y = y.to(self.device)  
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand())) 

            outs, feats = self.model.base(x)
            feats_cams = compute_fem_cam(feats, k=2, axis=(2, 3))
            feats = feats_cams
            if self.global_features is None:
                self.global_features = torch.zeros(
                    (self.train_samples, *feats.shape[1:]), device=self.device
                )
            soft_output = F.softmax(self.model.head(outs), dim=1).detach()  
            
            one_hot_y = F.one_hot(y.long(), self.num_classes).float()
            alpha = 0.8 * (self.current_round / self.max_rounds)  
            soft_y = (1 - alpha) * one_hot_y + alpha * soft_output   
    
            if self.current_round == 0:  
                       
                self.prev_soft_labels[idx] = soft_y
                self.global_features[idx] = feats
                
            else:
                ema_prev_features = self.global_features[idx]  
                self.global_features[idx] = (self.sigma) * ema_prev_features + (1 - self.sigma) * feats

            
        for param in self.model.base.parameters():
            param.requires_grad = True
        for param in self.model.head.parameters():
            param.requires_grad = True    
            
        for epoch in range(max_local_epochs):  
            for x, y, idx in trainloader:  
                if type(x) == type([]):  
                    x[0] = x[0].to(self.device)  
                else:
                    x = x.to(self.device)  
                y = y.to(self.device)  
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand())) 
                
                outs, rep = self.model.base(x)
                fep_cams = compute_fem_cam(rep, k=2, axis=(2, 3))
                rep = fep_cams
                
                output = self.model.head(outs)
                last_y = self.prev_soft_labels[idx]
                loss = self.loss(output, y) + (self.kl_divergence_loss(output, last_y)) * self.beta
                

                rep_g = self.global_features[idx]
                loss_mse = CAT_loss(rep, rep_g) * self.lamda
                loss += loss_mse
                
                self.optimizer.zero_grad()  
                loss.backward()  
                self.optimizer.step()  
                if epoch == max_local_epochs - 1:
                    soft_output = F.softmax(output, dim=1).detach()  
                    one_hot_y = F.one_hot(y.long(), self.num_classes).float()
                    alpha = (self.current_round / self.max_rounds)  
                    soft_y = (1 - alpha) * one_hot_y + alpha * soft_output 

                    ema_prev_soft_labels = self.prev_soft_labels[idx]
                    self.prev_soft_labels[idx] = soft_y 
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()  

        self.train_time_cost['num_rounds'] += 1  
        self.train_time_cost['total_cost'] += time.time() - start_time  


    def set_parameters(self, model):
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()


    def kl_divergence_loss(self, output, soft_output):
        log_output = F.log_softmax(output, dim=1)
        kl_loss = F.kl_div(log_output, soft_output, reduction='batchmean')
        return kl_loss
    

def CAT_loss(CAM_Student, CAM_Teacher, IF_NORMALIZE=False):
    loss = F.mse_loss(_Normalize(CAM_Student, IF_NORMALIZE), _Normalize(CAM_Teacher, IF_NORMALIZE))
    return loss


def _Normalize(feat, IF_NORMALIZE):
    if IF_NORMALIZE:
        feat = F.normalize(feat, dim=(2, 3))  
    return feat




def compute_fem_cam(activations, k, axis=(2, 3)):
    stds, means = torch.std_mean(activations, dim=axis, keepdim=True, unbiased=True)
    th = means + k * stds
    binary_mask = activations <= th
    weights = binary_mask.float().mean(dim=axis, keepdim=True).to(activations.dtype)
    fem_cam = weights * activations
    return fem_cam

