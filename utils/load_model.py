import os
import numpy as np
from PIL import Image
from tqdm import trange
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.models as models
import os
import numpy as np
from PIL import Image
from tqdm import trange
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights
class FineTuneResnet50(nn.Module):
    def __init__(self, num_class=1):
        super(FineTuneResnet50, self).__init__()
        self.num_class = num_class
        resnet50_net = models.resnet50(pretrained=True)
        # resnet50_net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # state_dict = torch.load("./models/resnet50-19c8e357.pth")
        # resnet50_net.load_state_dict(state_dict)
        self.features = nn.Sequential(*list(resnet50_net.children())[:-1])
        self.classifier = nn.Linear(2048, self.num_class)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x
    
class LeNet5(nn.Module):

    def __init__(self, num_classes=1, grayscale=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            # nn.Linear(400, 120), # cnc版本 3*32*32
            nn.Linear(16*4*4, 120), # 非cnc版本 3*28*28
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        if self.num_classes == 1:
            logits = self.sigmoid(logits)
        else:
            logits = F.softmax(logits,dim=1)
        return logits
        # probas = F.softmax(logits, dim=1)
        # return probas


# class LeNet5(nn.Module):
#     def __init__(self, num_classes):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16 * 5 * 5
#         self.fc2 = nn.Linear(120, 84)  # Activations layer
#         self.fc = nn.Linear(84, num_classes)
#         self.relu_1 = nn.ReLU()
#         self.relu_2 = nn.ReLU()
#         self.activation_layer = torch.nn.ReLU

#     def forward(self, x):
#         # Doing this way because only want to save activations
#         # for fc linear layers - see later
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1)#x.view(-1, 16 * 4 * 4)
#         x = self.relu_1(self.fc1(x))
#         x = self.relu_2(self.fc2(x))
#         x = self.fc(x)
#         return x


class SpuriousNet(nn.Module):
    def __init__(self, data_dim,hidden_size=10,output=1):
        super(SpuriousNet, self).__init__()
        self.fc1 = nn.Linear(data_dim, output)
        self.sigmoid = nn.Sigmoid()
        self.output = output
    def forward(self, x):
        x = self.fc1(x)
        if self.output == 1:
            x = self.sigmoid(x)
        else:
            x = F.softmax(x, dim=1)
        return x

class DISKNet(nn.Module):
    def __init__(self,data_dim=1, out_dim = 1, hidden_size=10):
        super(DISKNet, self).__init__()
        self.layers1 = nn.Sequential(nn.Linear(out_dim + data_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
        self.layers2 = nn.Sequential(nn.Linear(out_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self,z_tr,z_eval,D_tr,D_eval,avg_et, unbiased_loss):
        batch_size = z_tr.size(0)
        
        # KL(P(Y^tr,D)||q(Y^te,D))
        tiled_x = torch.cat([z_tr, z_eval], dim=0)
        concat_y = torch.cat([D_tr, D_eval], dim=0)
        # The upper part represents the joint distribution P(X,Y), while the lower part represents the marginal distribution PX and PY.
        inputs = torch.cat([tiled_x, concat_y], dim=1)

        logits = self.layers1(inputs)
        # the input of the first term in lower boudary P(X,Y)
        pred_xy = logits[:batch_size]
        # the input of the second term in lower boudary PX and PY
        pred_x_y = logits[batch_size:]
        
        # KL(P(D)||q(D))
        inputs_D = torch.cat([D_tr, D_eval], dim=0)
        logits_D = self.layers2(inputs_D)
        pred_xy_D = logits_D[:batch_size]
        pred_x_y_D = logits_D[batch_size:]
        
        if unbiased_loss:
            avg_et = 0.01 * avg_et + 0.99 * torch.mean(torch.mean(torch.exp(pred_x_y)))
            loss1  = np.log2(np.exp(1)) * (torch.mean(pred_xy)- torch.mean(torch.exp(pred_x_y)/avg_et).detach() * torch.log(torch.mean(torch.exp(pred_x_y))))
            loss2 = np.log2(np.exp(1)) * (torch.mean(pred_xy_D) - torch.mean(torch.exp(pred_x_y_D)/avg_et).detach() * torch.log(torch.mean(torch.exp(pred_x_y_D))))
            loss = loss1 - loss2
        else:
            loss1 = np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))) 
            loss2 = np.log2(np.exp(1)) * (torch.mean(pred_xy_D) - torch.log(torch.mean(torch.exp(pred_x_y_D))))
            loss = loss1 - loss2
        return loss,avg_et

class DISKNet_noy(nn.Module):
    def __init__(self,data_dim, out_dim = 1,hidden_size=10):
        super(DISKNet_noy, self).__init__()
        # self.avg_et=avg_et
        # self.convlayers=nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(3136, 64),
        #     nn.ReLU()
        # )
        # self.layers1 = nn.Sequential(nn.Linear(1 + data_dim, hidden_size),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, 1))
        # self.layers2 = nn.Sequential(nn.Linear(1, hidden_size),
        #                             nn.ReLU(),
        #                             nn.Linear(hidden_size, 1))
        self.layers1 = nn.Sequential(nn.Linear(out_dim + data_dim, 1))
        self.layers2 = nn.Sequential(nn.Linear(out_dim, 1))

    def forward(self,z_tr,z_eval,D_tr,D_eval,avg_et, unbiased_loss):
        batch_size = z_tr.size(0)
        tiled_x = torch.cat([z_tr, z_eval], dim=0)
        #z_tr   D_tr
        #z_eval D_eval
        concat_y = torch.cat([D_tr, D_eval], dim=0)
        # The upper part represents the joint distribution P(X,Y), while the lower part represents the marginal distribution PX and PY.
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers1(inputs)
        # the input of the first term in lower boudary P(X,Y)
        pred_xy = logits[:batch_size]
        # the input of the second term in lower boudary PX and PY
        pred_x_y = logits[batch_size:]
        
        
        # KL(P(D)||q(D))
        inputs_D = torch.cat([D_tr, D_eval], dim=0)
        logits_D = self.layers2(inputs_D)
        pred_xy_D = logits_D[:batch_size]
        pred_x_y_D = logits_D[batch_size:]
        
        if unbiased_loss:
            avg_et = 0.01 * avg_et + 0.99 * torch.mean(torch.mean(torch.exp(pred_x_y)))
            loss1  = np.log2(np.exp(1)) * (torch.mean(pred_xy)- torch.mean(torch.exp(pred_x_y)/avg_et).detach() * torch.log(torch.mean(torch.exp(pred_x_y))))
            loss2 = np.log2(np.exp(1)) * (torch.mean(pred_xy_D) - torch.mean(torch.exp(pred_x_y_D)/avg_et).detach() * torch.log(torch.mean(torch.exp(pred_x_y_D))))
            loss = loss1-loss2
        else:
            loss1 = np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))) 
            loss2 = np.log2(np.exp(1)) * (torch.mean(pred_xy_D) - torch.log(torch.mean(torch.exp(pred_x_y_D))))
            loss = loss1-loss2
        return loss,avg_et

from transformers import AlbertForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import DebertaV2ForSequenceClassification
import types
import torch
def _bert_replace_fc(model):
    model.fc = model.classifier
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)
    
    model.classifier = types.MethodType(classifier, model)

    model.base_forward = model.forward

    def forward(self, x):
        return self.base_forward(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits

    model.forward = types.MethodType(forward, model)
    return model
def bert_pretrained(output_dim):
	return _bert_replace_fc(BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=output_dim))
def bert(output_dim):
    config_class = BertForSequenceClassification.config_class
    config = config_class.from_pretrained(
            'bert-base-uncased', num_labels=output_dim)
    return _bert_replace_fc(BertForSequenceClassification(config))

import transformers
def bert_adamw_optimizer(model, lr, weight_decay):
    # Adapted from https://github.com/facebookresearch/BalancingGroups/blob/main/models.py
    # del momentum
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer

from tqdm import tqdm
def calculate_spurious_percentage(loader):
    label_spurious_count = {}  # 用于存储每个标签对应的spurious label 的数量
    label_count = {}  # 用于存储每个标签的总数量

    for inputs, labels, spurious in tqdm(loader, desc='Processing data', leave=True):
        # print(labels,spurious)
        for label, spurious_label in zip(labels, spurious):
            label = label.item()  # 将标签转换为Python标量
            spurious_label = spurious_label.item()  # 将spurious label 转换为Python标量

            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

            if label in label_spurious_count:
                if spurious_label in label_spurious_count[label]:
                    label_spurious_count[label][spurious_label] += 1
                else:
                    label_spurious_count[label][spurious_label] = 1
            else:
                label_spurious_count[label] = {spurious_label: 1}

    # 计算每个标签对应的spurious label 的百分比
    label_spurious_percentage = {}
    for label in label_spurious_count:
        total_count = label_count[label]
        spurious_percentage = {spurious_label: count for spurious_label, count in label_spurious_count[label].items()}
        # {spurious_label: count / total_count for spurious_label, count in label_spurious_count[label].items()}
        label_spurious_percentage[label] = spurious_percentage

    return label_spurious_percentage
