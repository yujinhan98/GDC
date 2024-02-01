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
from torch.utils.data import Subset,Dataset, DataLoader
import torch.utils.data as data
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import torch.optim as optim
import pandas as pd
from wilds.common.data_loaders import get_eval_loader
from sklearn.metrics import roc_auc_score, accuracy_score
import copy
from tqdm import tqdm
import transformers
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from collections import defaultdict
from spuco.models import model_factory 
from spuco.utils import Trainer
from torch.optim import SGD
from spuco.robust_train import GroupDRO
from spuco.models import model_factory 
from spuco.datasets import GroupLabeledDatasetWrapper
from torch.optim import SGD
from spuco.robust_train import UpSampleERM, DownSampleERM, CustomSampleERM
from spuco.models import model_factory 
from spuco.datasets import GroupLabeledDatasetWrapper
import time
sys.path.append('/home/yujin/dm/disk')
from utils.load_data import load_cmnist, load_waterbirds, load_celebA, load_civilcomments, load_multipy_cmnist,load_colored_mnist_cnc,load_mcolor
from utils.load_model import FineTuneResnet50,LeNet5,SpuriousNet,DISKNet,DISKNet_noy, bert_pretrained,bert_adamw_optimizer, calculate_spurious_percentage
from torchmetrics import Accuracy
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='The parameters of DISK')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--save_path' ,'-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=1)
# parser.add_argument('--r50', type=int, default=1)
# parser.add_argument('--sload',type=str, default=None, help='Folder to for extractor.')
parser.add_argument('--eload',type=str, default=None, help='Folder to for extractor.')
parser.add_argument('--dataset', type=str, default='cmnist', help='cmnist|waterbirds|celebA|muticmnist|mcolor')
args = parser.parse_args()
device = torch.device("cuda:{}".format(args.gpu))
args.save = args.save_path + args.dataset + '_metric_erm'
if os.path.isdir(args.save) == False:
    os.mkdir(args.save)
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
import re
model_loads_text = re.search(r'label(.*?)epoch', args.eload)
model_loads_text = model_loads_text.group(1)

if args.dataset == 'cmnist':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_cmnist()
    train_loader_img = DataLoader(
    train_dataset_img,
    batch_size=args.batch_size, shuffle=False, pin_memory=True,num_workers=16)
    eval_loader_img = DataLoader(
    eval_dataset_img,
    batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
    test_loader_img = DataLoader(
    test_dataset_img,
    batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=16)
elif args.dataset == 'mcolor':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_mcolor()
    train_loader_img = DataLoader(
    train_dataset_img,
    batch_size=args.batch_size, shuffle=False, pin_memory=True)
    eval_loader_img = DataLoader(
    eval_dataset_img,
    batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader_img = DataLoader(
    test_dataset_img,
    batch_size=args.batch_size, shuffle=False, pin_memory=True)
elif args.dataset == 'waterbirds':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_waterbirds()
    train_loader_img = get_eval_loader("standard", train_dataset_img, batch_size=args.batch_size, pin_memory=True,num_workers=16)
    eval_loader_img = get_eval_loader("standard",eval_dataset_img, batch_size=args.batch_size, pin_memory=True,num_workers=16)
    test_loader_img = get_eval_loader("standard", test_dataset_img, batch_size=args.batch_size, pin_memory=True,num_workers=16)
elif args.dataset == 'celebA':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_celebA()
    train_loader_img = get_eval_loader("standard", train_dataset_img, batch_size=args.batch_size, pin_memory=True,num_workers=16)
    eval_loader_img = get_eval_loader("standard",eval_dataset_img, batch_size=args.batch_size, pin_memory=True,num_workers=16)
    test_loader_img = get_eval_loader("standard", test_dataset_img, batch_size=args.batch_size, pin_memory=True,num_workers=16)
elif args.dataset == 'civilcomments':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_civilcomments()
    train_loader_img = DataLoader(train_dataset_img, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    # new_subset = Subset(train_loader_img.dataset, indices=[i for i in range(len(train_loader_img.dataset)) if i != 141037])
    # train_loader_img = DataLoader(new_subset, batch_size=train_loader_img.batch_size, shuffle=False, num_workers=8, pin_memory=train_loader_img.pin_memory)
    eval_loader_img = DataLoader(eval_dataset_img, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    test_loader_img = DataLoader(test_dataset_img, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    # new_subset = Subset(test_loader_img.dataset, indices=[i for i in range(len(test_loader_img.dataset)) if i != 73529])
    # test_loader_img = DataLoader(new_subset, batch_size=test_loader_img.batch_size, shuffle=False, num_workers=8, pin_memory=test_loader_img.pin_memory)
    train_dataset_img, eval_dataset_img,test_dataset_img =  train_loader_img.dataset,eval_loader_img.dataset,test_loader_img.dataset


def train_scratch(model, epochs):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3,momentum=0.9)
    start_time = time.time() 
    for epoch in range(epochs):
        running_loss = 0.0
        # for inputs, labels,_ in train_loader:
        for inputs, labels,_ in train_loader_img:
            optimizer.zero_grad()
            inputs, labels=inputs.to(device),labels.to(device)
            # print(inputs.shape)
            outputs = model(inputs)
            # print('outputs',outputs.shape,'labels.float().view(-1, 1)',labels.float().view(-1, 1).shape)
            loss = criterion(outputs, labels.float().view(-1, 1))  # 将标签的维度调整为 [batch_size, 1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # scheduler.step()
        # torch.save(model.state_dict(),
        #            os.path.join(args.save, args.dataset + '_scratch' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) +
        #                     '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
        # # Let us not waste space and delete the previous model
        # prev_path = os.path.join(args.save, args.dataset + '_scratch' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) +
        #                     '_' + args.mode+ '_epoch_' + str(epoch-1) + '.pt')
        # if os.path.exists(prev_path): os.remove(prev_path)
        end_time = time.time()  # 
        epoch_time = end_time - start_time 
        print('Epoch [%d/%d] | Time: %.2f seconds | Loss: %.4f' % (epoch+1,100,epoch_time , running_loss/len(train_loader_img)))
        start_time = time.time()
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
                                      f.write('%03d,%0.4f\n' % ((epoch + 1),running_loss/len(train_loader_img)))
    return model



#### Load Embeding

if args.dataset == 'cmnist': #r50:
    state_dict = torch.load(args.eload, map_location=device)
    net = LeNet5().to(device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
elif args.dataset == 'waterbirds':
    state_dict = torch.load(args.eload, map_location=device)
    net = FineTuneResnet50().to(device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    # net = FineTuneResnet50().to(device)
    # with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
    #        f.write('Training Scratch: epoch,train_loss(%)\n')
    # net = train_scratch(net, 100)
    # net.eval()
elif args.dataset == 'celebA':
    state_dict = torch.load(args.eload, map_location=device)
    net = FineTuneResnet50().to(device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
elif args.dataset == 'civilcomments':
    state_dict = torch.load(args.eload, map_location=device)
    net = bert_pretrained(output_dim=2).to(device)
    net.load_state_dict(state_dict)
    net.eval()

def test(data_loader_img):
    net.eval()
    correct = 0
    scorrect = 0
    pred_labels = []
    spu_labels = []
    true_labels = []
    with torch.no_grad():
        for data_emb, label, slabel in tqdm(data_loader_img):
            data_emb, label,slabel = data_emb.to(device), label.to(device), slabel.to(device)#.cuda()
            if args.dataset == 'waterbirds' or args.dataset == 'celebA':
                slabel = slabel[:, 0]
            output = net(data_emb)
            # print(output)
            # pred = output.data.max(1)[1]
            if len(output.shape) > 1:
                pred = torch.argmax(output, dim=1)
            else:
                pred = torch.squeeze(torch.round(torch.squeeze(output)))
            pred_labels.extend(pred.cpu().numpy())
            # pred_probs.extend(torch.squeeze(output).cpu().numpy())
            spu_labels.extend(slabel.cpu().numpy())
            true_labels.extend(label.cpu().numpy())
            correct += (pred == label).sum().item()#pred.eq(label.data).sum().item()
            scorrect += (pred == slabel).sum().item()#pred.eq(slabel.data).sum().item()

    accuracy = correct / len(data_loader_img.dataset)
    saccuracy = scorrect / len(data_loader_img.dataset)
    if saccuracy < 0.5:
        pred_labels = [1-value for value in pred_labels]
    saccuracy = max(saccuracy,1-saccuracy)
    return accuracy, saccuracy, pred_labels, true_labels,spu_labels





# extract embedding
with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
    f.write(f'Starting Predict...\n')
print('Beginning Predicting\n')
with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
       f.write('DISK: train_accuracy,eval_accuracy,test_accuracy\n')
train_accuracy, train_saccuracy, train_slabels, train_labels,spu_train_labels= test(train_loader_img)
eval_accuracy, eval_saccuracy,eval_slabels,eval_labels,spu_eval_labels  = test(eval_loader_img)
test_accuracy, test_saccuracy,test_slabels,test_labels,spu_test_labels = test(test_loader_img)
print('Train Accuracy: %.4f | Eval Accuracy: %.4f | Test Accuracy: %.4f' % (train_accuracy,eval_accuracy,test_accuracy))
print('Train SAccuracy: %.4f | Eval SAccuracy: %.4f | Test SAccuracy: %.4f' % (train_saccuracy,eval_saccuracy,test_saccuracy))
with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
        f.write('%0.4f,%0.4f,%0.4f\n' % (train_accuracy,eval_accuracy,test_accuracy))

with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
     f.write('DISK: train_saccuracy,eval_saccuracy,test_saccuracy\n')

with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
    f.write('%0.4f,%0.4f,%0.4f\n' % (train_saccuracy,eval_saccuracy,test_saccuracy))



if args.dataset == 'cmnist':
    worst_indices_train = [i for i in range(len(train_labels)) if train_labels[i] == spu_train_labels[i]]#[idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == slabel]
    worst_indices_eval = [i for i in range(len(eval_labels)) if eval_labels[i] == spu_eval_labels[i]]#[idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == slabel]
    worst_indices_test = [i for i in range(len(test_labels)) if test_labels[i] == spu_test_labels[i]]#[idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == slabel]

elif args.dataset == 'waterbirds':
    worst_indices_train = [i for i in range(len(train_labels)) if train_labels[i] != spu_train_labels[i]]#[idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label != slabel[0]]
    worst_indices_eval = [i for i in range(len(eval_labels)) if eval_labels[i] != spu_eval_labels[i]]#[idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label != slabel[0]]
    worst_indices_test = [i for i in range(len(test_labels)) if test_labels[i] != spu_test_labels[i]]#[idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label != slabel[0]]
elif args.dataset == 'celebA':
    worst_indices_train = [i for i in range(len(train_labels)) if train_labels[i] == spu_train_labels[i] and train_labels[i]==1]#[idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == 1 and slabel[0] == 1]
    worst_indices_eval = [i for i in range(len(eval_labels)) if eval_labels[i] == spu_eval_labels[i] and eval_labels[i]==1]#[idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == 1 and slabel[0] == 1]
    worst_indices_test = [i for i in range(len(test_labels)) if test_labels[i] == spu_test_labels[i] and test_labels[i]==1]#[idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == 1 and slabel[0] == 1]

from collections import defaultdict
def recall_precision(labels,slabels,spu_labels,worst_indices):
    if args.dataset == 'cmnist':
        pre_minority = [i for i in range(len(slabels)) if slabels[i] == labels[i] ]
        correct = [i for i in range(len(slabels)) if slabels[i] == spu_labels[i] and spu_labels[i] == labels[i] ]
        precision = len(correct) / len(pre_minority) * 100
        recall = len(correct) / len(worst_indices) * 100
        print(f"Precision-Percentage of common elements in predicted_minority_group: {precision}%")
        print(f"Recall-Percentage of common elements in minority_group: {recall}%")
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
            f.write('\nMinority Group Precision: %0.4f\n'%(precision))
            f.write('\nMinority Group Recall: %0.4f\n'%(recall))
    elif args.dataset == 'waterbirds':
        pre_minority = [i for i in range(len(slabels)) if slabels[i] != labels[i] ]
        correct = [i for i in range(len(slabels)) if slabels[i] == spu_labels[i] and spu_labels[i] != labels[i] ]
        precision = len(correct) / len(pre_minority) * 100
        recall = len(correct) / len(worst_indices) * 100
        print(f"Precision-Percentage of common elements in predicted_minority_group: {precision}%")
        print(f"Recall-Percentage of common elements in minority_group: {recall}%")
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
            f.write('\nMinority Group Precision: %0.4f\n'%(precision))
            f.write('\nMinority Group Recall: %0.4f\n'%(recall))
    elif args.dataset == 'celebA':
        worst_indices = [i for i in range(len(slabels)) if spu_labels[i] == 1  and labels[i] == 1]
        pre_minority = [i for i in range(len(slabels)) if slabels[i] == 1 and labels[i]==1 ]
        correct = [i for i in range(len(slabels)) if slabels[i] == spu_labels[i] and spu_labels[i] == 1  and labels[i] == 1 ]
        # print(len(correct),len(pre_minority),len(worst_indices))
        if len(pre_minority)!=0:
            precision = len(correct) / len(pre_minority) * 100
        else:
            precision = 0
        recall = len(correct) / len(worst_indices) * 100
        print(f"Precision-Percentage of common elements in predicted_minority_group: {precision}%")
        print(f"Recall-Percentage of common elements in minority_group: {recall}%")
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
            f.write('\nMinority Group Precision: %0.4f\n'%(precision))
            f.write('\nMinority Group Recall: %0.4f\n'%(recall))
    elif args.dataset == 'civilcomments':
        key = (0,0)
        print('The group is:',key)
        worst_indices = [i for i in range(len(slabels)) if spu_labels[i] == key[0] and labels[i]==key[1]]
        pre_minority = [i for i in range(len(slabels)) if slabels[i] == key[0] and labels[i]==key[1]]
        correct = [i for i in range(len(slabels)) if slabels[i] == spu_labels[i] and spu_labels[i] == key[0] and labels[i]==key[1] ]
        if len(pre_minority)!=0:
            precision = len(correct) / len(pre_minority) * 100
        else:
            precision = 0
        recall = len(correct) / len(worst_indices) * 100
        print(f"The Group is :(%.1f,%.1f) | Precision-Percentage of common elements in predicted_minority_group: {key[0],key[1],precision}%")
        print(f"The Group is :(%.1f,%.1f) | Recall-Percentage of common elements in minority_group: {key[0],key[1],recall}%")
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
            f.write('The Group is :(%.1f,%.1f) | Minority Group Precision: %0.4f\n'%(key[0],key[1],precision))
            f.write('The Group is :(%.1f,%.1f) | Minority Group Recall: %0.4f\n'%(key[0],key[1],recall))
        key = (1,0)
        print('The group is:',key)
        worst_indices = [i for i in range(len(slabels)) if spu_labels[i] == key[0] and labels[i]==key[1]]
        pre_minority = [i for i in range(len(slabels)) if slabels[i] == key[0] and labels[i]==key[1]]
        correct = [i for i in range(len(slabels)) if slabels[i] == spu_labels[i] and spu_labels[i] == key[0] and labels[i]==key[1] ]
        if len(pre_minority)!=0:
            precision = len(correct) / len(pre_minority) * 100
        else:
            precision = 0
        recall = len(correct) / len(worst_indices) * 100
        print(f"The Group is :(%.1f,%.1f) | Precision-Percentage of common elements in predicted_minority_group: {key[0],key[1],precision}%")
        print(f"The Group is :(%.1f,%.1f) | Recall-Percentage of common elements in minority_group: {key[0],key[1],recall}%")
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
            f.write('The Group is :(%.1f,%.1f) | Minority Group Precision: %0.4f\n'%(key[0],key[1],precision))
            f.write('The Group is :(%.1f,%.1f) | Minority Group Recall: %0.4f\n'%(key[0],key[1],recall))
        key = (0,1)
        print('The group is:',key)
        worst_indices = [i for i in range(len(slabels)) if spu_labels[i] == key[0] and labels[i]==key[1]]
        pre_minority = [i for i in range(len(slabels)) if slabels[i] == key[0] and labels[i]==key[1]]
        correct = [i for i in range(len(slabels)) if slabels[i] == spu_labels[i] and spu_labels[i] == key[0] and labels[i]==key[1] ]
        if len(pre_minority)!=0:
            precision = len(correct) / len(pre_minority) * 100
        else:
            precision = 0
        recall = len(correct) / len(worst_indices) * 100
        print(f"The Group is :(%.1f,%.1f) | Precision-Percentage of common elements in predicted_minority_group: {key[0],key[1],precision}%")
        print(f"The Group is :(%.1f,%.1f) | Recall-Percentage of common elements in minority_group: {key[0],key[1],recall}%")
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
            f.write('The Group is :(%.1f,%.1f) | Minority Group Precision: %0.4f\n'%(key[0],key[1],precision))
            f.write('The Group is :(%.1f,%.1f) | Minority Group Recall: %0.4f\n'%(key[0],key[1],recall))
        key = (1,1)
        print('The group is:',key)
        worst_indices = [i for i in range(len(slabels)) if spu_labels[i] == key[0] and labels[i]==key[1]]
        pre_minority = [i for i in range(len(slabels)) if slabels[i] == key[0] and labels[i]==key[1]]
        correct = [i for i in range(len(slabels)) if slabels[i] == spu_labels[i] and spu_labels[i] == key[0] and labels[i]==key[1] ]
        if len(pre_minority)!=0:
            precision = len(correct) / len(pre_minority) * 100
        else:
            precision = 0
        recall = len(correct) / len(worst_indices) * 100
        print(f"The Group is :(%.1f,%.1f) | Precision-Percentage of common elements in predicted_minority_group: {key[0],key[1],precision}%")
        print(f"The Group is :(%.1f,%.1f) | Recall-Percentage of common elements in minority_group: {key[0],key[1],recall}%")
        with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
            f.write('The Group is :(%.1f,%.1f) | Minority Group Precision: %0.4f\n'%(key[0],key[1],precision))
            f.write('The Group is :(%.1f,%.1f) | Minority Group Recall: %0.4f\n'%(key[0],key[1],recall))
    #     # break
        
    # indices_dict = defaultdict(list)
    # for idx, (label, slabel) in enumerate(zip(labels, slabels)):
    #     key = (label, slabel)
    #     indices_dict[key].append(idx)
    # for key, indices in indices_dict.items():
    #     print(f"Indices for key {key}: {len(indices)}")
    # # count_00 = len(indices_dict.get((0, 0), []))
    # # count_01 = len(indices_dict.get((0, 1), []))
    # # count_11 = len(indices_dict.get((1, 1), []))
    # # count_10 = len(indices_dict.get((1, 0), []))
    # if args.dataset != 'civilcomments':
    #     # if count_00 + count_11 > count_10 + count_01:
    #     #     keys = [(1,0),(0,1)]
    #     # else:
    #     #     keys = [(0,0),(1,1)]
    #     if args.dataset == 'cmnist':
    #         keys = [(1,1),(0,0)]
    #     elif args.dataset == 'waterbirds':
    #         keys = [(1,0),(0,1)]
    #     elif args.dataset == 'celebA':
    #         keys = [min(indices_dict, key=lambda k: len(indices_dict[k]))]
    #     print("keys is:",keys)
    #     with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
    #                 f.write('The Mni Group is :%s,'%(str(keys)))
    #     minority_indices = []
    #     for key, indices in indices_dict.items():
    #         if key in keys:
    #             print(f"selected Indices for key {key}: {len(indices)}")
    #             minority_indices += indices_dict[key]
    # elif args.dataset == 'celebA':
    #     min_key = min(indices_dict, key=lambda k: len(indices_dict[k]))
    #     minority_indices = []
    #     key = (1,1)
    #     for key, indices in indices_dict.items():
    #         if key == min_key:
    #             print(f"selected Indices for key {key}: {len(indices)}")
    #             minority_indices = indices_dict[key]
    # elif args.dataset == 'celebA':
    #     min_key = min(indices_dict, key=lambda k: len(indices_dict[k]))
    #     minority_indices = []
    #     for key, indices in indices_dict.items():
    #         if key == min_key:
    #             print(f"selected Indices for key {key}: {len(indices)}")
    #             minority_indices = indices_dict[key]
    # elif args.dataset == 'civilcomments':
    #     for key, indices in indices_dict.items():
    #         print(f"selected Indices for key {key}: {len(indices)}")
    #         minority_indices = indices_dict[key]
    #         worst_indices = [i for i in range(len(labels)) if labels[i] == key[0] and slabels[i]==key[1]]
    #         common_indices = set(minority_indices) & set(worst_indices)
    #         percentage_common_minority = len(common_indices) / len(minority_indices) * 100
    #         percentage_common_worst = len(common_indices) / len(worst_indices) * 100
    #         print(f"The Group is :(%.1f,%.1f) | Precision-Percentage of common elements in predicted_minority_group: {key[0],key[1],percentage_common_minority}%")
    #         print(f"The Group is :(%.1f,%.1f) | Recall-Percentage of common elements in minority_group: {key[0],key[1],percentage_common_worst}%")
    #         with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
    #             f.write('The Group is :(%.1f,%.1f) | Minority Group Precision: %0.4f\n'%(key[0],key[1],percentage_common_minority))
    #             f.write('The Group is :(%.1f,%.1f) | Minority Group Recall: %0.4f\n'%(key[0],key[1],percentage_common_worst))
    #     # break
    # if args.dataset != 'civilcomments':
    #     common_indices = set(minority_indices) & set(worst_indices)
    #     percentage_common_minority = len(common_indices) / len(minority_indices) * 100
    #     percentage_common_worst = len(common_indices) / len(worst_indices) * 100
    #     print(f"Precision-Percentage of common elements in predicted_minority_group: {percentage_common_minority}%")
    #     print(f"Recall-Percentage of common elements in minority_group: {percentage_common_worst}%")
    #     with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
    #         f.write('\nMinority Group Precision: %0.4f\n'%(percentage_common_minority))
    #         f.write('\nMinority Group Recall: %0.4f\n'%(percentage_common_worst))
if args.dataset == 'civilcomments':
    worst_indices_train,worst_indices_eval,worst_indices_test = [],[],[]

with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
       f.write('Training\n')
recall_precision(train_labels,train_slabels,spu_train_labels,worst_indices_train)
with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
       f.write('Validation\n')
recall_precision(eval_labels,eval_slabels,spu_eval_labels,worst_indices_eval)
with open(os.path.join(args.save, args.dataset+ '_s' + str(args.seed)+ '_smodel'+str(model_loads_text)+'_metric.csv'), 'a') as f:
       f.write('Test\n')
recall_precision(test_labels,test_slabels,spu_test_labels,worst_indices_test)


