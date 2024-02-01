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
import time
sys.path.append('/home/yujin/dm/GDC')
from utils.load_data import load_cmnist, load_waterbirds, load_celebA, load_civilcomments
from utils.load_model import FineTuneResnet50,LeNet5,SpuriousNet,GDCNet,GDCNet_noy, bert_pretrained,bert_adamw_optimizer
from torchmetrics import Accuracy
import warnings
from collections import defaultdict
from spuco.models import model_factory 
from spuco.datasets import GroupLabeledDatasetWrapper
from torch.optim import SGD
from spuco.robust_train import UpSampleERM, DownSampleERM, CustomSampleERM
from spuco.models import model_factory 
from spuco.datasets import GroupLabeledDatasetWrapper
from spuco.utils import Trainer
from torch.optim import SGD
from spuco.robust_train import GroupDRO
from spuco.datasets import GroupLabeledDatasetWrapper

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='The parameters of GDC')


parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--epochs','-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
parser.add_argument('--decay','-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--momomentum', type=float, default=0.9, help='momomentum.')
parser.add_argument('--GDC_epochs', type=int, default=5, help='Number of epochs to train GDC.')
parser.add_argument('--GDC_lr', type=float, default=1e-5, help='The initial learning rate of GDC.')
parser.add_argument('--use_label', action='store_false', help='Use validation label or not')
# parser.add_argument('--adjust', action='store_true', help='adjust eval or not')
parser.add_argument('--eval_weight', type=int, default=2, help='Loss weight of GDC')
# parser.add_argument('--cluster_lr', type=float, default=1e-4, help='Description of cluster learning rate')
# parser.add_argument('--hcs', type=float, default=None, help='High confidence sampling threshold')
parser.add_argument('--upweight_factor', type=str, default='auto', help='10|20|100|auto')
parser.add_argument('--save_path' ,'-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=2) 
# parser.add_argument('--r50', type=int, default=1)
# parser.add_argument('--mode', type=str, default='groupdro', choices = ['upweight','downweight','mixweight','groupdro'],help='Training Mode')
parser.add_argument('--model_type', choices=['r50', 'bert', 'scratch-r50','lenet5'], default='bert')
parser.add_argument('--load',type=str, default=None, help='Folder to for extractor.')
parser.add_argument('--sload',type=str, default=None, help='Folder for spurious.')
parser.add_argument('--dataset', type=str, default='civilcomments', help='cmnist|waterbirds|celebA|civilcomments')
# parser.add_argument('--es', type=str, default='train', help='None|train|eval')
# parser.add_argument('--mode', type=str, default='downweight', help='upweight|downweight|groupdro|jtt')
# parser.add_argument('--lambdas', type=int, default=6, help='lambda')
args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))


args.save = args.save_path + args.dataset + args.model_type+'_epoch'+str(args.epochs)+'_lr'+str(args.lr) + '_es' +str(args.es) + '_neq_'+str(args.mode)+'_new'
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

if args.dataset == 'cmnist':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_cmnist()
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
    train_loader_img = get_eval_loader("standard", train_dataset_img, batch_size=args.batch_size)
    eval_loader_img = get_eval_loader("standard",eval_dataset_img, batch_size=args.batch_size)
    test_loader_img = get_eval_loader("standard", test_dataset_img, batch_size=args.batch_size)
elif args.dataset == 'celebA':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_celebA()
    train_loader_img = get_eval_loader("standard", train_dataset_img, batch_size=args.batch_size)
    eval_loader_img = get_eval_loader("standard",eval_dataset_img, batch_size=args.batch_size)
    test_loader_img = get_eval_loader("standard", test_dataset_img, batch_size=args.batch_size)
elif args.dataset == 'civilcomments':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_civilcomments()
    train_loader_img = DataLoader(train_dataset_img, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    eval_loader_img = DataLoader(eval_dataset_img, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    test_loader_img = DataLoader(test_dataset_img, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    train_dataset_img, eval_dataset_img,test_dataset_img =  train_loader_img.dataset,eval_loader_img.dataset,test_loader_img.dataset




def train_scratch(model, epochs):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
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
        torch.save(model.state_dict(),
                   os.path.join(args.save, args.dataset + '_scratch' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset + '_scratch' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch-1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, args.epochs, running_loss/len(train_loader_img)))
        with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%03d,%0.4f\n' % ((epoch + 1),running_loss/len(train_loader_img)))
    return model



#Create model

if args.model_type == 'r50': #r50:
    net = FineTuneResnet50().to(device)
elif args.model_type == 'scratch-r50':
    net = FineTuneResnet50().to(device)
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)
elif args.model_type == 'lenet5': #r50:
    net = LeNet5().to(device)
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)
elif args.model_type == 'bert':
    net = bert_pretrained(output_dim=2).to(device)




# /////////////// Training ///////////////
criterion_train = torch.nn.CrossEntropyLoss()
def train(train_loader_img,min_loader_img=None):
    net.train()
    correct = 0
    best_acc = 0
    best_epoch = 0
    total = 0
    if args.model_type == 'r50':
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        optimizer_train = optim.SGD(net.classifier.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    elif args.model_type == 'bert':
        optimizer_train = bert_adamw_optimizer(net, lr=args.lr, weight_decay=args.decay)
        # optim.SGD(net.classifier.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    elif args.model_type == 'scratch-r50' or args.model_type == 'lenet5':
        optimizer_train = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    start_time = time.time()  
    for epoch in range(args.epochs):
        running_loss = 0.0
        for (inputs, labels,_) in train_loader_img:
            optimizer_train.zero_grad()
            inputs, labels=inputs.to(device),labels.to(device)
            # print(inputs.shape)
            outputs = net(inputs)
            if args.model_type == 'bert':
                loss = criterion_train(outputs, labels)
            else:
                loss = criterion_train(outputs, labels.float().view(-1, 1))
            optimizer_train.zero_grad()
            loss.backward()
            optimizer_train.step()
            preds = torch.argmax(outputs, dim=1)
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # acc = (preds == labels).float().mean()
            running_loss += loss.item()
        # scheduler.step()
        acc = correct / total
        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time 
        print('Epoch [%d/%d] | Time: %.2f seconds | Train Loss: %.4f | Train ACC: %.4f ' % (epoch + 1, args.epochs, epoch_time, running_loss / len(train_loader_img), train_accuracy))
        start_time = time.time()
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch-1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)

        if min_loader_img is not None:
            min_correct = 0
            min_total = 0
            with torch.no_grad():
                for images, labels, _ in min_loader_img:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    pred = torch.argmax(outputs, dim=1)
                    min_total += labels.size(0)
                    min_correct += (pred == labels).sum().item()
            min_acc = min_correct / min_total
            print(f'Accuracy on min_loader_img: {min_acc}')
            if min_acc > best_acc:
                prev_path = os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_best_epoch'  + '.pt')
                if os.path.exists(prev_path): os.remove(prev_path)
                best_acc = min_acc
                best_epoch = epoch
                torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_best_epoch' + '.pt'))
                print(f'Saved best model with accuracy: {best_acc} at epoch {best_epoch+1}\n')
                with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                    f.write(f'Saved best model with accuracy: {best_acc} at epoch {best_epoch+1}\n')
            
            with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%03d,%0.4f, %.4f, %.4f\n' % ((epoch + 1),running_loss/len(train_loader_img),acc,min_acc))


                      
        


def test():
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target,_ in test_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            # forward
            output = net(data)
            preds = torch.argmax(output, dim=1)
            if len(target.shape) > 1:
                target = torch.argmax(target, dim=1)
            correct += (preds == target).sum().item()
    test_accuracy = correct / len(test_loader_img.dataset)

    correct = 0
    with torch.no_grad():
        for data, target,_ in train_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            preds = torch.argmax(output, dim=1)
            if len(target.shape) > 1:
                target = torch.argmax(target, dim=1)
            correct += (preds == target).sum().item()
    train_accuracy = correct / len(train_loader_img.dataset)

    correct = 0
    with torch.no_grad():
        for data, target,_ in eval_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            preds = torch.argmax(output, dim=1)
            if len(target.shape) > 1:
                target = torch.argmax(target, dim=1)
            correct += (preds == target).sum().item()
    eval_accuracy = correct / len(eval_loader_img.dataset)


    print('Train Accuracy: %.4f | Eval Accuracy: %.4f |Test Accuracy: %.4f ' % (train_accuracy,eval_accuracy, test_accuracy))
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
         f.write('Train Accuracy| Eval Accuracy|Test Accuracy: %0.4f,%0.4f,%0.4f\n' % (train_accuracy,eval_accuracy, test_accuracy))
    
def test_worst(values):
    if args.dataset == 'cmnist':
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == slabel]
    elif args.dataset == 'waterbirds':
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label != slabel[0]]
    elif args.dataset == 'celebA':
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == 1 and slabel[0] == 1]
    elif args.dataset == 'civilcomments':
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == values[0] and slabel == values[1]]
    
    worst_dataset = Subset(test_loader_img.dataset, worst_indices_test)
    worst_loader = DataLoader(worst_dataset, batch_size= args.batch_size, shuffle=True)
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target,_ in worst_loader:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            preds = torch.argmax(output, dim=1)
            if len(target.shape) > 1:
                target = torch.argmax(target, dim=1)
            correct += (preds == target).sum().item()
    accuracy = correct / len(worst_loader.dataset)
    print('The Group is :(%.1f,%.1f) | Test Worst Accuracy: %.4f ' % (values[0],values[1],accuracy))
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
         f.write('The Group is :(%.1f,%.1f) | Test Worst Group Accuracy: %0.4f\n' % (values[0],values[1],accuracy))
    return accuracy

if args.model_type == 'scratch-r50':
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch:\n')
    test()
    test_worst()
elif args.model_type == 'lenet5': #r50:
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch:\n')
    test()
    test_worst()


class Spucodataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.group_weights = self.calculate_group_weights()
        self.group_partition = self.create_group_partition()
        self.labels = [item[1] for item in data]
        self.num_classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        label = self.data[idx][1]
        return image, label
    def calculate_group_weights(self):
        group_counts = defaultdict(int)
        total_samples = len(self.data)
        # print(total_samples)
        for item in self.data:
            item2=item[2]
            group = (item[1].item(), item2.item())  # Assuming the group information is at index 2
            group_counts[group] += 1
        # print(group_counts)
        group_weights = {group: count / total_samples for group, count in group_counts.items()}
        return group_weights
    def create_group_partition(self):
        group_partition = defaultdict(list)
        for idx, item in enumerate(self.data):
            item2=item[2]
            group = (item[1].item(), item2.item())  # Using a tuple of (label, group) as the key
            group_partition[group].append(idx)
        return dict(group_partition) 


print('Generate Eval Dataset')
evalset = Spucodataset(eval_dataset_img)
print('Generate Train Dataset')
trainset = Spucodataset(train_dataset_img)
print('Generate Test Dataset')
testset = Spucodataset(test_dataset_img)

# /////////////// Extracting ///////////////

if args.model_type == 'r50':
    model = FineTuneResnet50().to(device)
elif args.model_type == 'scratch-r50':
    model = copy.deepcopy(net)
elif args.model_type == 'lenet5':
    model = copy.deepcopy(net)
elif args.model_type == 'bert':
    model = copy.deepcopy(net)
if args.model_type == 'r50' or args.model_type == 'scratch-r50' or args.model_type == 'lenet5':
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    feature_extractor[0].fc = torch.nn.Identity()
elif args.model_type == 'bert':
    feature_extractor = copy.deepcopy(net)
    feature_extractor.fc = torch.nn.Identity()



class EmbDataset(Dataset):
    def __init__(self, data_dir, labels, slabels):
        self.data_dir = data_dir
        self.labels = labels
        self.slabels = slabels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = torch.load(os.path.join(self.data_dir, f'emb_{idx}.pt'))
        label = self.labels[idx]
        slabel = self.slabels[idx]
        return emb, label, slabel

def embbeding(data_loader_img,name):
    emb_dir = 'emb_data/'+'label'+str(args.use_label)+'_seed'+str(args.seed)+'_'+name+'_civil_emb_data'
    os.makedirs(emb_dir, exist_ok=True)
    labels_list = []
    slabels_list = []
    for images, labels, slabels in tqdm(data_loader_img, desc="Extracting"):
        images, labels, slabels = images.to(device), labels.to(device), slabels.to(device)
        if args.dataset == 'waterbirds' or args.dataset == 'celebA':
            slabels = slabels[:, 0]
        outputs = feature_extractor(images).squeeze()
        for i, emb in enumerate(outputs):
            torch.save(emb, os.path.join(emb_dir, f'emb_{len(labels_list) + i}.pt'))
        labels_list.extend(labels.tolist())
        slabels_list.extend(slabels.tolist())

    dataset = EmbDataset(emb_dir, labels_list, slabels_list)
    emb_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return emb_loader


def train_GDC(train_loader, eval_loader,start_avg_et=1.0,unbiased_loss=False):
    # all_mi = []
    # CE = []
    # MINE = []
    avg_et = start_avg_et
    for epoch in range(args.GDC_epochs):
        running_loss = 0.0
        kl_loss = 0.0
        pre_loss = 0.0
        for (inputs_train, labels_train, _), (inputs_eval, labels_eval, _) in zip(train_loader, eval_loader):
            inputs_train = inputs_train.to(device)
            labels_train = labels_train.to(device)
            inputs_eval = inputs_eval.to(device)
            labels_eval = labels_eval.to(device)

            optimizer_spurious.zero_grad()
            optimizer_GDC.zero_grad()

            outputs_train = spurious_net(inputs_train)
            outputs_eval = spurious_net(inputs_eval)

            # Prediction Loss
            loss_cluster_train = nn.BCELoss()(outputs_train, labels_train.float().view(-1, 1))

            # MI Loss
            if args.use_label:
                KL_loss,avg_et = GDC_net(labels_train.float().view(-1, 1),labels_eval.float().view(-1, 1),outputs_train,outputs_eval,avg_et,unbiased_loss=unbiased_loss)
            else:
                KL_loss,avg_et = GDC_net(inputs_train, inputs_eval, outputs_train, outputs_eval,avg_et,unbiased_loss=unbiased_loss)
            
            # Final Loss
            loss = loss_cluster_train - args.eval_weight * KL_loss

            loss.backward()
            optimizer_spurious.step()
            optimizer_GDC.step()

            kl_loss = KL_loss.item()
            pre_loss = loss_cluster_train.item()
            running_loss = loss.item()

        print('Epoch [%d/%d]| Train Loss: %.4f | KL Loss: %.4f | Prediction Loss: %.4f' % (epoch+1, args.GDC_epochs, running_loss, kl_loss, pre_loss))

        torch.save(spurious_net.state_dict(),
               os.path.join(args.save, args.dataset + '_spurious' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset + '_spurious' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)
        with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%03d,%0.4f,%0.4f,%0.4f\n' % ((epoch + 1),running_loss,kl_loss,pre_loss))


def test_GDC(data_loader_img,hcs=args.hcs):
    spurious_net.eval()
    correct = 0
    scorrect = 0
    pred_labels = []
    # pred_probs = []
    true_labels = []
    with torch.no_grad():
        for data_emb, label,slabel in data_loader_img:
            data_emb, label,slabel = data_emb.to(device), label.to(device), slabel.to(device)#.cuda()
            output = spurious_net(data_emb)
            # print(output)
            # pred = output.data.max(1)[1]
            pred = torch.squeeze(torch.round(torch.squeeze(output)))
            if hcs is not None:
                for i in range(len(pred)):
                    if pred[i] != label[i] and hcs < output[i] < 1-hcs:
                           pred[i] = label[i]
            pred_labels.extend(pred.cpu().numpy())
            # pred_probs.extend(torch.squeeze(output).cpu().numpy())
            true_labels.extend(label.cpu().numpy())
            correct += (pred == label).sum().item()#pred.eq(label.data).sum().item()
            scorrect += (pred == slabel).sum().item()#pred.eq(slabel.data).sum().item()

    accuracy = correct / len(data_loader_img.dataset)
    saccuracy = scorrect / len(data_loader_img.dataset)
    saccuracy = max(saccuracy,1-saccuracy)
    return accuracy, saccuracy, pred_labels, true_labels



train_loader_emb =  embbeding(train_loader_img,'train')
eval_loader_emb =  embbeding(eval_loader_img,'eval')
test_loader_emb =  embbeding(test_loader_img,'test')

if args.adjust:
    dim = train_loader_emb.dataset[0][0].shape[0]
    spurious_net = SpuriousNet(data_dim = dim).to(device)
    state_dict = torch.load(args.sload, map_location=device)
    spurious_net.load_state_dict(state_dict)
    spurious_net.to(device)
    eval_accuracy, eval_saccuracy,eval_slabels,eval_labels = test_GDC(eval_loader_emb)
    from collections import defaultdict
    indices_dict = defaultdict(list)
    for idx, (label, slabel) in enumerate(zip(eval_labels, eval_slabels)):
        key = (label, slabel)
        indices_dict[key].append(idx)
    for key, indices in indices_dict.items():
          print(f"Eval Indices for key {key}: {len(indices)}")
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs) + '_dlr' + str(args.GDC_lr) + '_label' + str(args.use_label) + '_hcs' + str(args.hcs) + '_s' + str(args.seed) + '_' + args.mode + str(args.upweight_factor) + '_training_results.csv'), 'a') as f:
        f.write('Eval Group Info:\n')
        for key, indices in indices_dict.items():
            f.write(f"Indices for key {key}: {len(indices)}\n")
            f.write(f"Key {key} has {len(indices)} samples before balancing\n")
    min_samples = sorted([len(indices) for indices in indices_dict.values()])[::-1][1]  # Get the second smallest sample size
    balanced_indices = []
    for key, indices in indices_dict.items():
        if len(indices) < min_samples:
            ratio = int(min_samples / len(indices))
            repeated_indices = [idx for idx in indices for _ in range(ratio)]
            indices_dict[key] = repeated_indices
        else:
            indices_dict[key] = indices
        print(f"Key {key} has {len(indices_dict[key])} samples after balancing")
    with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs) + '_dlr' + str(args.GDC_lr) + '_label' + str(args.use_label) + '_hcs' + str(args.hcs) + '_s' + str(args.seed) + '_' + args.mode + str(args.upweight_factor) + '_training_results.csv'), 'a') as f:
        f.write('Eval Group Info:\n')
        for key, indices in indices_dict.items():
            f.write(f"Indices for key {key}: {len(indices)}\n")
            f.write(f"Key {key} has {len(indices)} samples after balancing\n")
    original_dataset = eval_loader_img.dataset
    upsampled_indices = []
    for indices in indices_dict.values():
        upsampled_indices.extend(indices)
    upsampled_dataset = data.Subset(original_dataset, upsampled_indices)
    eval_loader_img = data.DataLoader(upsampled_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    eval_loader_emb =  embbeding(eval_loader_img,'eval')



# /////////////// Training GDC ///////////////

dim = train_loader_emb.dataset[0][0].shape[0]
spurious_net = SpuriousNet(data_dim = dim).to(device)
if args.use_label:
    GDC_net = GDCNet().to(device)
else:
    GDC_net = GDCNet_noy(data_dim = dim).to(device)

optimizer_spurious =  optim.SGD(spurious_net.parameters(), lr=args.GDC_lr, momentum=0.9)
optimizer_GDC = optim.SGD(GDC_net.parameters(),  lr=args.GDC_lr, momentum=0.9)


with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    for key, value in state.items():
        f.write('%s:%s\n' % (key, value))
with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    f.write('epoch,train_loss,KL_loss,pre_error(%)\n')

print('Beginning Training GDC\n')

train_GDC(train_loader_emb, eval_loader_emb)

print('Beginning Testing GDC\n')
with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('GDC: train_accuracy,eval_accuracy,test_accuracy\n')
train_accuracy, train_saccuracy, train_slabels, train_labels= test_GDC(train_loader_emb)
eval_accuracy, eval_saccuracy,eval_slabels,eval_labels = test_GDC(eval_loader_emb)
test_accuracy, test_saccuracy,_,_ = test_GDC(test_loader_emb)
print('Train Accuracy: %.4f | Eval Accuracy: %.4f | Test Accuracy: %.4f' % (train_accuracy,eval_accuracy,test_accuracy))
print('Train SAccuracy: %.4f | Eval SAccuracy: %.4f | Test SAccuracy: %.4f' % (train_saccuracy,eval_saccuracy,test_saccuracy))
with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%0.4f,%0.4f,%0.4f\n' % (train_accuracy,eval_accuracy,test_accuracy))

with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('GDC: train_saccuracy,eval_saccuracy,test_saccuracy\n')

with open(os.path.join(args.save, args.dataset + '_GDC' + '_depoch' + str(args.GDC_epochs)+'_dlr'+str(args.GDC_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%0.4f,%0.4f,%0.4f\n' % (train_saccuracy,eval_saccuracy,test_saccuracy))