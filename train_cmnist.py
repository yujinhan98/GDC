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
sys.path.append('/home/yujin/dm/disk')
from utils.load_data import load_cmnist, load_waterbirds, load_celebA, load_civilcomments, load_multipy_cmnist,load_colored_mnist_cnc,load_mcolor
from utils.load_model import FineTuneResnet50,LeNet5,SpuriousNet,DISKNet,DISKNet_noy, bert_pretrained,bert_adamw_optimizer, calculate_spurious_percentage
from torchmetrics import Accuracy
from torch.utils.data import TensorDataset, DataLoader
parser = argparse.ArgumentParser(description='The parameters of DISK')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs','-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--retrain', type=int, default=20, help='Number of epochs to retrain.')
parser.add_argument('--lr', type=float, default=1e-3, help='The initial learning rate.')
parser.add_argument('--decay','-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--momomentum', type=float, default=0.9, help='momomentum.')
parser.add_argument('--disk_epochs', type=int, default=20, help='Number of epochs to train disk.')
parser.add_argument('--disk_lr', type=float, default=1e-5, help='The initial learning rate of disk.')
parser.add_argument('--use_label', action='store_false', help='Use validation label or not')
parser.add_argument('--eval_weight', type=int, default=10, help='Loss weight of disk')
# parser.add_argument('--cluster_lr', type=float, default=1e-4, help='Description of cluster learning rate')
parser.add_argument('--hcs', type=float, default=None, help='High confidence sampling threshold')
parser.add_argument('--mode', type=str, default='jtt', help='upweight|downweight|groupdro|jtt')
parser.add_argument('--upweight_factor', type=str, default='auto', help='10|20|100|auto')
parser.add_argument('--save_path' ,'-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=1)
# parser.add_argument('--r50', type=int, default=1)
parser.add_argument('--model_type', choices=['r50', 'bert', 'scratch-r50','lenet5'], default='lenet5')
parser.add_argument('--load',type=str, default=None, help='Folder to for extractor.')
parser.add_argument('--dataset', type=str, default='cmnist', help='cmnist|waterbirds|celebA|muticmnist|mcolor')
parser.add_argument('--es', type=str, default='train', help='None|train|eval')
parser.add_argument('--lambdas', type=int, default=5, help='lambda')
args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))


args.save = args.save_path + args.dataset + args.model_type+'_epoch'+str(args.epochs)+'_lr'+str(args.lr) + '_es' +str(args.es)+'_'+str(args.mode)+'_group_new'
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
elif args.dataset == 'muticmnist':
    # colored_mnist_train, colored_mnist_val, colored_mnist_test  = load_colored_mnist_cnc(args)
    def load_multipy_cmnist_loader(dataset):
           dataset_img = TensorDataset(torch.stack([torch.tensor(x) for x in dataset.data.X]), 
                                  torch.stack([torch.tensor(x) for x in dataset.data.labels]), 
                                  torch.stack([torch.tensor(x) for x in dataset.data.spurious]))
           loader_img = DataLoader(dataset_img, batch_size=args.batch_size, shuffle=False)
           return loader_img
    trainset, valset,testset  = load_multipy_cmnist()
    # def load_multipy_cmnist_loader(dataset):
    #       dataset_img = TensorDataset(torch.stack([torch.tensor(x) for x in dataset.data]), 
    #                               torch.stack([torch.tensor(x) for x in dataset.targets_all['target']]), 
    #                               torch.stack([torch.tensor(x) for x in dataset.targets_all['spurious']]))
    #       loader_img = DataLoader(dataset_img, batch_size=args.batch_size, shuffle=False)
    #       return loader_img
    # trainset, valset,testset  = load_colored_mnist_cnc(args)
    train_loader_img = load_multipy_cmnist_loader(trainset)
    eval_loader_img = load_multipy_cmnist_loader(valset)
    test_loader_img = load_multipy_cmnist_loader(testset)

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
            if args.dataset == 'muticmnist':
                loss = torch.nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
            else:
                loss = criterion(outputs, labels.float().view(-1, 1))  # 将标签的维度调整为 [batch_size, 1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # scheduler.step()
        torch.save(model.state_dict(),
                   os.path.join(args.save, args.dataset + '_scratch' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset + '_scratch' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch-1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)
        print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, args.epochs, running_loss/len(train_loader_img)))
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label)+'_hcs'+str(args.hcs) + '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%03d,%0.4f\n' % ((epoch + 1),running_loss/len(train_loader_img)))
    return model



#Create model

if args.model_type == 'r50': #r50:
    net = FineTuneResnet50().to(device)
elif args.model_type == 'scratch-r50':
    net = FineTuneResnet50().to(device)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)
elif args.model_type == 'lenet5' and args.dataset == 'cmnist': #r50:
    net = LeNet5().to(device)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)
elif args.model_type == 'lenet5' and args.dataset == 'mcolor': #r50:
    net = LeNet5().to(device)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)
elif args.model_type == 'lenet5' and args.dataset == 'muticmnist':
    net = LeNet5(num_classes=5).to(device)
    # print('num=5')
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)
       



# /////////////// Training ///////////////
criterion_train = nn.BCELoss()#torch.nn.CrossEntropyLoss()

def train(train_loader_img,min_loader_img=None):
    net.train()
    correct = 0
    best_acc = 0
    min_acc = 0
    best_epoch = 0
    total = 0
    if args.model_type == 'r50':
        for name, param in net.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        optimizer_train = optim.SGD(net.classifier.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    else:
        optimizer_train = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for (inputs, labels,_) in train_loader_img:
            optimizer_train.zero_grad()
            inputs, labels=inputs.to(device),labels.to(device)
            # print(inputs.shape)
            outputs = net(inputs)
            if args.dataset == 'muticmnist':
                loss = torch.nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
            else:
                loss = criterion_train(outputs, labels.float().view(-1, 1))
            optimizer_train.zero_grad()
            loss.backward()
            optimizer_train.step()
            running_loss += loss.item()
            if args.dataset == 'muticmnist':
                preds = outputs.data.max(1)[1]
            else:
                preds = torch.squeeze(torch.round(torch.squeeze(outputs)))
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        # scheduler.step()
        train_accuracy = correct / total
        print('Epoch [%d/%d] | Train Loss: %.4f | Train ACC: %.4f' % (epoch+1, args.epochs, running_loss/len(train_loader_img),train_accuracy))

        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch-1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)

        # Show results
        # with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
        #                               f.write('%03d,%0.4f\n' % ((epoch + 1),running_loss/len(train_loader_img)))

        if min_loader_img is not None:
            min_correct = 0
            min_total = 0
            with torch.no_grad():
                for images, labels, _ in min_loader_img:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    if args.dataset == 'muticmnist':
                        pred = outputs.data.max(1)[1]
                    else:
                        pred = torch.squeeze(torch.round(torch.squeeze(outputs)))
                    # pred = torch.squeeze(torch.round(torch.squeeze(outputs)))
                    min_total += labels.size(0)
                    min_correct += (pred == labels).sum().item()
            min_acc = min_correct / min_total
            print(f'Accuracy on min_loader_img: {min_acc}')
            if min_acc > best_acc:
                prev_path = os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_best_epoch'  + '.pt')
                if os.path.exists(prev_path): os.remove(prev_path)
                best_acc = min_acc
                best_epoch = epoch
                torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_best_epoch' + '.pt'))
                print(f'Saved best model with accuracy: {best_acc} at epoch {best_epoch+1}\n')
                with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write(f'Saved best model with accuracy: {best_acc} at epoch {best_epoch}\n')
        
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%03d,%0.4f, %.4f, %.4f\n' % ((epoch + 1),running_loss/len(train_loader_img),train_accuracy,min_acc))

    

def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target,_ in test_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            # forward
            output = net(data)
            # accuracy
            # pred = torch.argmax(output, dim=1)
            # pred = torch.squeeze(torch.round(torch.squeeze(output)))
            if args.dataset == 'muticmnist':
                pred = torch.argmax(output, dim=1)
            else:
                pred = torch.squeeze(torch.round(torch.squeeze(output)))
            total += target.size(0)
            correct += (pred == target).sum().item()
    test_accuracy = correct / total

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target,_ in train_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            # pred = torch.argmax(output, dim=1)
            # pred = torch.squeeze(torch.round(torch.squeeze(output)))
            if args.dataset == 'muticmnist':
                pred = torch.argmax(output, dim=1)
            else:
                pred = torch.squeeze(torch.round(torch.squeeze(output)))
            correct += (pred == target).sum().item()
            total += target.size(0)
    train_accuracy = correct / total

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target,_ in eval_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            # pred = torch.argmax(output, dim=1)
            # pred = torch.squeeze(torch.round(torch.squeeze(output)))
            if args.dataset == 'muticmnist':
                pred = torch.argmax(output, dim=1)
            else:
                pred = torch.squeeze(torch.round(torch.squeeze(output)))
            correct += (pred == target).sum().item()
            total += target.size(0)
    eval_accuracy = correct / total


    print('Train Accuracy: %.4f | Eval Accuracy: %.4f |Test Accuracy: %.4f ' % (train_accuracy,eval_accuracy, test_accuracy))
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
         f.write('Train Accuracy| Eval Accuracy|Test Accuracy: %0.4f,%0.4f,%0.4f\n' % (train_accuracy,eval_accuracy, test_accuracy))
    
def test_worst():
    if args.dataset == 'cmnist':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == slabel]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == slabel]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == slabel]
    elif args.dataset == 'mcolor':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == slabel]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == slabel]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == slabel]
    elif args.dataset == 'muticmnist':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == slabel]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == slabel]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == values[0] and slabel == values[1]]
    elif args.dataset == 'waterbirds':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label != slabel[0]]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label != slabel[0]]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label != slabel[0]]
    elif args.dataset == 'celebA':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == 1 and slabel[0] == 1]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == 1 and slabel[0] == 1]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == 1 and slabel[0] == 1]
    
    worst_dataset = Subset(test_loader_img.dataset, worst_indices_test)
    worst_loader = DataLoader(worst_dataset, batch_size= args.batch_size, shuffle=True)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target,_ in worst_loader:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            if args.dataset == 'muticmnist':
                pred = torch.argmax(output, dim=1)#output.data.max(1)[1]
            else:
                pred = torch.squeeze(torch.round(torch.squeeze(output)))
            # pred = torch.squeeze(torch.round(torch.squeeze(output)))
            # pred = torch.argmax(output, dim=1)
            correct += pred.eq(target.data).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print('Test Worst Accuracy: %.4f ' % (accuracy))
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
         f.write('Test Worst Group Accuracy: %0.4f\n' % (accuracy))
    # print('The Group is :(%.1f,%.1f) | Test Worst Accuracy: %.4f ' % (values[0],values[1],accuracy))
    # with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    #      f.write('The Group is :(%.1f,%.1f) | Test Worst Group Accuracy: %0.4f\n' % (values[0],values[1],accuracy))
    return accuracy

def test_worst_(values):
    if args.dataset == 'cmnist':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == slabel]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == slabel]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == values[0] and slabel == values[1]]
    elif args.dataset == 'waterbirds':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label != slabel[0]]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label != slabel[0]]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label != slabel[0]]
    elif args.dataset == 'celebA':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == 1 and slabel[0] == 1]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == 1 and slabel[0] == 1]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == 1 and slabel[0] == 1]
    elif args.dataset == 'civilcomments':
        # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == 1 and slabel[0] == 1]
        # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == 1 and slabel[0] == 1]
        worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == values[0] and slabel == values[1]]
    
    worst_dataset = Subset(test_loader_img.dataset, worst_indices_test)
    worst_loader = DataLoader(worst_dataset, batch_size= args.batch_size, shuffle=True)
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target,_ in worst_loader:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            preds = torch.squeeze(torch.round(torch.squeeze(output)))
            # print(preds,target)
            if len(target.shape) > 1:
                target = torch.argmax(target, dim=1)
            correct += (preds == target).sum().item()
    accuracy = correct / len(worst_loader.dataset)
    print('The Group is :(%.1f,%.1f) | The sample size is:  %.0f |Test Worst Accuracy: %.4f ' % (values[0],values[1],len(worst_indices_test),accuracy))
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
         f.write('The Group is :(%.1f,%.1f) | Test Worst Group Accuracy: %0.4f\n' % (values[0],values[1],accuracy))
    return accuracy


if args.model_type == 'scratch-r50':
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch:\n')
    test()
    test_worst()
    test_worst_([0,1])
    test_worst_([1,0])
    test_worst_([0,0])
    test_worst_([1,1])
elif args.model_type == 'lenet5': #r50:
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch:\n')
    test()
    test_worst()
#     worst_acc = 1
#     for i in range(1):
#         for j in range(1):
#             group_acc = test_worst([i, j])
#             if group_acc < worst_acc:
#                   worst_acc = group_acc
#                   values0, values1 = i,j
# with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
#          f.write('The Worst Group is :(%.1f,%.1f) | Test Worst Group Accuracy: %0.4f\n' % (values0,values1,worst_acc))
                 

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
            group = (item[1], int(item[2]))  # Assuming the group information is at index 2
            group_counts[group] += 1
        # print(group_counts)
        group_weights = {group: count / total_samples for group, count in group_counts.items()}
        return group_weights
    def create_group_partition(self):
        group_partition = defaultdict(list)
        for idx, item in enumerate(self.data):
            group = (item[1], int(item[2]))  # Using a tuple of (label, group) as the key
            group_partition[group].append(idx)
        return dict(group_partition) 




# /////////////// Extracting ///////////////

if args.model_type == 'r50':
    model = FineTuneResnet50().to(device)
elif args.model_type == 'scratch-r50':
    model = copy.deepcopy(net)
elif args.model_type == 'lenet5':
    model = copy.deepcopy(net)
# else:
#     # exteactor model load
#     model = ResNet_Model(name='resnet34', num_classes=num_classes)
#     model = model.to(device)

feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
feature_extractor[0].fc = torch.nn.Identity()

class EmbDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(data.iloc[:, :-2].values, dtype=torch.float32)
        self.labels = torch.tensor(data.iloc[:, -2].values, dtype=torch.float32)
        self.slabels = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        slabel = self.slabels[index]
        return feature, label, slabel

def embbeding(data_loader_img):
    fc_outputs = []
    labels_list = []
    slabels_list = []
    print('Beginning Extracting\n')
    for images, labels, slabels in data_loader_img:
        images, labels=images.to(device),labels.to(device)
        # print(images.shape)
        if args.dataset=='waterbirds':
            slabels = slabels[:, 0]
        outputs = feature_extractor(images)
        # print('outputs:',outputs.shape)
        fc_output = outputs.view(outputs.size(0), -1).cpu().detach().numpy()
        # print('fc_outputs:',fc_outputs.shape)
        fc_outputs.append(fc_output)
        labels_list.append(labels.cpu().numpy())
        slabels_list.append(slabels.cpu().numpy())
    
    all_fc_outputs = np.concatenate(fc_outputs, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_slabels = np.concatenate(slabels_list, axis=0)
    all_data = pd.concat([pd.DataFrame(all_fc_outputs),  pd.DataFrame(all_labels),  pd.DataFrame(all_slabels)], axis=1)
    num_columns = all_data.shape[1]
    column_names = ['F' + str(i) for i in range(1, num_columns-1)]
    column_names.append('label')
    column_names.append('slabel')
    all_data.columns = column_names
    emb_loader = DataLoader(EmbDataset(all_data), batch_size=args.batch_size, shuffle=False)
    return emb_loader

# extract embedding

train_loader_emb =  embbeding(train_loader_img)
eval_loader_emb =  embbeding(eval_loader_img)
test_loader_emb =  embbeding(test_loader_img)

# /////////////// Training DISK ///////////////

dim = train_loader_emb.dataset[0][0].shape[0]
print(dim)
spurious_net = SpuriousNet(data_dim = dim).to(device)
if args.use_label:
    disk_net = DISKNet().to(device)
else:
    disk_net = DISKNet_noy(data_dim = dim).to(device)

optimizer_spurious =  optim.SGD(spurious_net.parameters(), lr=args.disk_lr, momentum=0.9)
optimizer_disk = optim.SGD(disk_net.parameters(),  lr=args.disk_lr, momentum=0.9)

def train_disk(train_loader, eval_loader,start_avg_et=1.0,unbiased_loss=False):
    # all_mi = []
    # CE = []
    # MINE = []
    avg_et = start_avg_et
    for epoch in range(args.disk_epochs):
        running_loss = 0.0
        kl_loss = 0.0
        pre_loss = 0.0
        for (inputs_train, labels_train, _), (inputs_eval, labels_eval, _) in zip(train_loader, eval_loader):
            inputs_train = inputs_train.to(device)
            labels_train = labels_train.to(device)
            inputs_eval = inputs_eval.to(device)
            labels_eval = labels_eval.to(device)

            optimizer_spurious.zero_grad()
            optimizer_disk.zero_grad()
            # print('inputs_train:',inputs_train.shape)

            outputs_train = spurious_net(inputs_train)
            # outputs_train = outputs_train.data.max(1)[1]
            outputs_eval = spurious_net(inputs_eval)
            # outputs_eval = outputs_eval.data.max(1)[1]

            # Prediction Loss
            if args.dataset == 'muticmnist':
                labels_train = labels_train.long()
                labels_eval = labels_eval.long()
                # loss_cluster_train = F.cross_entropy(outputs_train, labels_train)
                loss_cluster_train = nn.CrossEntropyLoss(reduction='mean')(outputs_train, labels_train)
            else:
                loss_cluster_train = nn.BCELoss()(outputs_train, labels_train.float().view(-1, 1))

            # MI Loss
            if args.use_label:
                KL_loss,avg_et = disk_net(labels_train.float().view(-1, 1),labels_eval.float().view(-1, 1),outputs_train,outputs_eval,avg_et,unbiased_loss=unbiased_loss)
            else:
                KL_loss,avg_et = disk_net(inputs_train, inputs_eval, outputs_train, outputs_eval,avg_et,unbiased_loss=unbiased_loss)
            
            # Final Loss
            loss = loss_cluster_train - args.eval_weight * KL_loss

            loss.backward()
            optimizer_spurious.step()
            optimizer_disk.step()

            kl_loss = KL_loss.item()
            pre_loss = loss_cluster_train.item()
            running_loss = loss.item()

        print('Epoch [%d/%d]| Train Loss: %.4f | KL Loss: %.4f | Prediction Loss: %.4f' % (epoch+1, args.disk_epochs, running_loss, kl_loss, pre_loss))

        torch.save(spurious_net.state_dict(),
               os.path.join(args.save, args.dataset + '_spurious' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset + '_spurious' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%03d,%0.4f,%0.4f,%0.4f\n' % ((epoch + 1),running_loss,kl_loss,pre_loss))


def test_disk(data_loader_img,hcs=args.hcs):
    spurious_net.eval()
    correct = 0
    scorrect = 0
    pred_labels = []
    # pred_probs = []
    true_labels = []
    spu_labels = []
    with torch.no_grad():
        for data_emb, label,slabel in data_loader_img:
            data_emb, label,slabel = data_emb.to(device), label.to(device), slabel.to(device)#.cuda()
            output = spurious_net(data_emb)
            # print(output)
            # pred = torch.argmax(output, dim=1)
            # pred = torch.squeeze(torch.round(torch.squeeze(output)))
            if args.dataset == 'muticmnist':
                pred = torch.argmax(output, dim=1)
            else:
                pred = torch.squeeze(torch.round(torch.squeeze(output)))
            if hcs is not None:
                for i in range(len(pred)):
                    if pred[i] != label[i] and hcs < output[i] < 1-hcs:
                           pred[i] = label[i]
            pred_labels.extend(pred.cpu().numpy())
            # pred_probs.extend(torch.squeeze(output).cpu().numpy())
            true_labels.extend(label.cpu().numpy())
            spu_labels.extend(slabel.cpu().numpy())
            correct += (pred == label).sum().item()#pred.eq(label.data).sum().item()
            scorrect += (pred == slabel).sum().item()#pred.eq(slabel.data).sum().item()

    # accuracy = adjusted_rand_score(true_labels, pred_labels)
    # saccuracy = adjusted_rand_score(spu_labels, pred_labels)
    accuracy = correct / len(data_loader_img.dataset)
    saccuracy = scorrect / len(data_loader_img.dataset)
    # if saccuracy < 0.5 and args.mode == 'jtt':
    #       pred_labels = [1-value for value in pred_labels]
    saccuracy = max(saccuracy,1-saccuracy)
    # ars = adjusted_rand_score(spu_labels, pred_labels)
    return accuracy, saccuracy, pred_labels, true_labels, spu_labels

with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    for key, value in state.items():
        f.write('%s:%s\n' % (key, value))
with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    f.write('epoch,train_loss,KL_loss,pre_error(%)\n')

print('Beginning Training DISK\n')

train_disk(train_loader_emb, eval_loader_emb)

print('Beginning Testing DISK\n')
with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('DISK TrueRand: train_truerand,eval_truerand,test_truerand\n')
train_accuracy, train_saccuracy, train_slabels, train_labels,train_spu_labels= test_disk(train_loader_emb)
eval_accuracy, eval_saccuracy,eval_slabels,eval_labels,eval_spu_labels = test_disk(eval_loader_emb)
test_accuracy, test_saccuracy,_,_,_ = test_disk(test_loader_emb)
print('Train TrueRand: %.4f | Eval TrueRand: %.4f | Test TrueRand: %.4f' % (train_accuracy,eval_accuracy,test_accuracy))
print('Train SpuRand: %.4f | Eval SpuRand: %.4f | Test SpuRand: %.4f' % (train_saccuracy,eval_saccuracy,test_saccuracy))
# print('Train ARS %.4f | Eval ARS: %.4f | Test ARS: %.4f' % (train_ars,eval_ars,test_ars))
with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%0.4f,%0.4f,%0.4f\n' % (train_accuracy,eval_accuracy,test_accuracy))

with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('DISK SpuRand: train_spurand,eval_spurand,test_spurand\n')

with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%0.4f,%0.4f,%0.4f\n' % (train_saccuracy,eval_saccuracy,test_saccuracy))


# with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
#                                       f.write('DISK: train_ars,eval_ars,test_ars\n')

# with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
#                                       f.write('%0.4f,%0.4f,%0.4f\n' % (train_ars,eval_ars,test_ars))


# /////////////// Retraining///////////////

from collections import defaultdict

indices_dict = defaultdict(list)

for idx, (label, slabel) in enumerate(zip(train_labels, train_slabels)):
    key = (label, slabel)
    indices_dict[key].append(idx)

if args.mode == 'groupdro' or args.mode == 'jtt':
      disk_group_partition = dict(indices_dict)

for key, indices in indices_dict.items():
    print(f"Indices for key {key}: {len(indices)}")

with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    f.write('Group Info:\n')
    for key, indices in indices_dict.items():
        f.write(f"Key {key} has {len(indices)} samples before balancing\n")
if args.es == 'train':
    min_group = min(indices_dict, key=lambda k: len(indices_dict[k]))

if args.mode == 'upweight':
      max_samples = max(len(indices) for indices in indices_dict.values())
      balanced_indices = []
      for key, indices in indices_dict.items():
            ratio = int(max_samples / len(indices))
            repeated_indices = [idx for idx in indices for _ in range(ratio)]
            indices_dict[key] = repeated_indices
      for key, indices in indices_dict.items():
            print(f"Key {key} has {len(indices)} samples after balancing")
            
      with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write('Predicted Train Group Info:\n')
            for key, indices in indices_dict.items():
                  f.write(f"Indices for key {key}: {len(indices)}\n")
                  f.write(f"Key {key} has {len(indices)} samples after balancing\n")



elif args.mode == 'downweight':
    min_samples = min(len(indices) for indices in indices_dict.values())
    balanced_indices = []
    for key, indices in indices_dict.items():
        if len(indices) > min_samples:
            sampled_indices = random.sample(indices, min_samples)
            indices_dict[key] = sampled_indices
        print(f"Key {key} has {len(indices_dict[key])} samples after balancing")

    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs) + '_dlr' + str(args.disk_lr) + '_label' + str(args.use_label) + '_hcs' + str(args.hcs) + '_s' + str(args.seed) + '_' + args.mode + str(args.upweight_factor) + '_training_results.csv'), 'a') as f:
        f.write('Group Info:\n')
        for key, indices in indices_dict.items():
            f.write(f"Indices for key {key}: {len(indices)}\n")
            f.write(f"Key {key} has {len(indices)} samples after balancing\n")

elif args.mode == 'jtt':
    count_00 = len(indices_dict.get((0, 0), []))
    count_01 = len(indices_dict.get((0, 1), []))
    count_11 = len(indices_dict.get((1, 1), []))
    count_10 = len(indices_dict.get((1, 0), []))
    if count_00 + count_11 > count_10 + count_01:
          keys = [(1,0),(0,1)]
    else:
          keys = [(0,0),(1,1)]
    # min_samples = min(len(indices) for indices in indices_dict.values())
    # min_samples = next((indices for indices in group_len if indices > min_samples), min_samples)
    # keys = [(1,1),(0,0)]
    balanced_indices = []
    for key, indices in indices_dict.items():
        if key in keys:
            ratio = args.lambdas + 1
            repeated_indices = [idx for idx in indices for _ in range(ratio)]
            indices_dict[key] = repeated_indices
            #   oversampled_indices = random.choices(indices, k=min_samples - len(indices))  # Upsample
            #   indices_dict[key].extend(oversampled_indices)
        print(f"Key {key} has {len(indices_dict[key])} samples after balancing")
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
              f.write('Group Info:\n')
              for key, indices in indices_dict.items():
                    f.write(f"Indices for key {key}: {len(indices)}\n")
                    f.write(f"Key {key} has {len(indices)} samples after balancing\n")




# max_samples = max(len(indices) for indices in indices_dict.values())//2
# balanced_indices = []
# for key, indices in indices_dict.items():
#     if len(indices) < max_samples:
#         repeated_indices = indices * (max_samples // len(indices)) + indices[:max_samples % len(indices)]
#     else:
#         repeated_indices = indices[:max_samples]
#     indices_dict[key] = repeated_indices


# for key, indices in indices_dict.items():
#     print(f"Key {key} has {len(indices)} samples after balancing")

# with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
#     f.write('Predicted Train Group Info:\n')
#     for key, indices in indices_dict.items():
#         f.write(f"Indices for key {key}: {len(indices)}\n")
#         f.write(f"Key {key} has {len(indices)} samples after balancing\n")

if args.es is not None:
    if args.es == 'eval':
        eval_indices_dict = defaultdict(list)
        for idx, (label, slabel) in enumerate(zip(eval_labels, eval_slabels)):
            key = (label, slabel)
            eval_indices_dict[key].append(idx)
        
        for key, indices in eval_indices_dict.items():
            print(f"Indices for eval key {key}: {len(indices)}")  
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write('Predicted Eval Group Info:\n')
            for key, indices in eval_indices_dict.items():
                f.write(f"Eval Key {key} has {len(indices)} samples before balancing\n")
        min_group = min(eval_indices_dict, key=lambda k: len(eval_indices_dict[k]))
        min_eval_samples = len(eval_indices_dict[min_group])
        min_group_indices = eval_indices_dict[min_group]
        print(f"The eval group with the minimum number of samples is {min_group} with {min_eval_samples} samples.")
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write(f"The eval group with the minimum number of samples is {min_group} with {min_eval_samples} samples.")
        eval_mingroup = data.Subset(eval_loader_img.dataset, min_group_indices)
        min_loader_img = data.DataLoader(eval_mingroup, batch_size=args.batch_size, shuffle=True)
    elif args.es == 'train':
        # After is better than before
        # min_group = min(indices_dict, key=lambda k: len(indices_dict[k]))
        min_train_samples = len(indices_dict[min_group])
        min_group_indices = indices_dict[min_group]
        print(f"The train group with the minimum number of samples is {min_group} with {min_train_samples} samples.")
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write(f"The train group with the minimum number of samples is {min_group} with {min_train_samples} samples.")
        train_mingroup = data.Subset(train_loader_img.dataset, min_group_indices)
        min_loader_img = data.DataLoader(train_mingroup, batch_size=args.batch_size, shuffle=True)
if args.es == None:
    min_loader_img = None
        

        # min_eval_samples = max(len(indices) for indices in eval_indices_dict.values())


if args.mode in  ['upweight','downweight','mixweight','jtt']:
    original_dataset = train_loader_img.dataset
    upsampled_indices = []
    for indices in indices_dict.values():
        upsampled_indices.extend(indices)
    # upsampled_indices = [item for item in minority_indices for _ in range(args.upweight_factor)]
    # upsampled_indices.extend([item for item in range(len(original_dataset)) if item not in minority_indices])
    upsampled_dataset = data.Subset(original_dataset, upsampled_indices)
    upsampled_train_loader_img = data.DataLoader(upsampled_dataset, batch_size=args.batch_size, shuffle=True)
    spurious_percentage = calculate_spurious_percentage(upsampled_train_loader_img)
    print("actual_spurious_percentage:",spurious_percentage)
    args.epochs = args.retrain
    train(upsampled_train_loader_img,min_loader_img)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
          f.write("Actual Spurious Percentage after sampling:\n")
          for key, value in spurious_percentage.items():
                f.write(f"{key}: {value}\n")
    
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write('Final Results: \n')   
    test()
    test_worst()
    test_worst_([0,1])
    test_worst_([1,0])
    test_worst_([0,0])
    test_worst_([1,1])
    # worst_acc = 1
    # for i in range(1):
    #     for j in range(1):
    #         group_acc = test_worst([i, j])
    #         if group_acc < worst_acc:
    #             worst_acc = group_acc
    #             values0, values1 = i,j
    # with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    #      f.write('The Worst Group is :(%.1f,%.1f) | Test Worst Group Accuracy: %0.4f\n' % (values0,values1,worst_acc))
    # test_worst()
    if args.es is not None:
        state_dict = torch.load(os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_best_epoch' + '.pt'), map_location=device)
        if args.model_type == 'r50': 
              net = FineTuneResnet50().to(device)
        elif args.model_type == 'lenet5' and args.dataset == 'cmnist': 
            net = LeNet5().to(device)
        elif args.model_type == 'lenet5' and args.dataset == 'mcolor': 
            net = LeNet5().to(device)
        elif args.model_type == 'lenet5' and args.dataset == 'muticmnist':
              net = LeNet5(num_classes=5).to(device)
        elif args.model_type == 'bert': 
            net = bert_pretrained(output_dim=2).to(device)
        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()    
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write('Best Results: \n')   
        test()
        test_worst()
        test_worst_([0,1])
        test_worst_([1,0])
        test_worst_([0,0])
        test_worst_([1,1])
        # worst_acc = 1
        # for i in range(1):
        #     for j in range(1):
        #           group_acc = test_worst([i, j])
        #           if group_acc < worst_acc:
        #                 worst_acc = group_acc
        #                 values0, values1 = i,j
        # with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
        #  f.write('The Worst Group is :(%.1f,%.1f) | Test Worst Group Accuracy: %0.4f\n' % (values0,values1,worst_acc))



elif args.mode ==  'groupdro':
    trainset = Spucodataset(train_dataset_img)
    evalset = Spucodataset(eval_dataset_img)
    testset = Spucodataset(test_dataset_img)

    group_labeled_trainset = GroupLabeledDatasetWrapper(trainset, disk_group_partition)
    if args.dataset == 'cmnist':
          model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)
          optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momomentum, nesterov=True)
    elif args.dataset in ['waterbirds','celebA']:
          model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes).to(device)
          optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momomentum, nesterov=True)
    elif args.dataset== 'civilcommnets':
          model = model_factory("distilbert", (0,0,0), trainset.num_classes).to(device)
          optimizer = bert_adamw_optimizer(model, lr=args.lr, weight_decay=args.decay)

    group_dro = GroupDRO(
            model=model,
            num_epochs=args.epochs,
            trainset=group_labeled_trainset,
            batch_size=args.batch_size,
            optimizer=optimizer,
            device=device,
            verbose=True
            )
    group_dro.train()
    from spuco.evaluate import Evaluator
    evaluator = Evaluator(
            testset=testset,
            group_partition=testset.group_partition,
            group_weights=trainset.group_weights,
            batch_size=args.batch_size,
            model=model,
            device=device,
            verbose=True)
    evaluator.evaluate()
    torch.save(model.state_dict(),
                   os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
                            '_' + args.mode+ '_epoch_' + str(args.epochs) + '.pt'))
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write('\nworst acc | average acc:\n')
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%0.4f,%0.4f\n' % (evaluator.worst_group_accuracy[1],evaluator.average_accuracy))
    print(evaluator.worst_group_accuracy)
    print(evaluator.average_accuracy)
                  

# elif args.mode == 'jtt':
#     trainset = Spucodataset(train_dataset_img)
#     evalset = Spucodataset(eval_dataset_img)
#     testset = Spucodataset(test_dataset_img)
#     group_labeled_trainset = GroupLabeledDatasetWrapper(trainset, disk_group_partition)
#     if args.dataset == 'cmnist':
#           model = model_factory("lenet", trainset[0][0].shape, trainset.num_classes).to(device)
#           optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momomentum, nesterov=True)
#     elif args.dataset in ['waterbirds','celebA']:
#           model = model_factory("resnet50", trainset[0][0].shape, trainset.num_classes).to(device)
#           optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momomentum, nesterov=True)
#     elif args.dataset== 'civilcommnets':
#           model = model_factory("distilbert", (0,0,0), trainset.num_classes).to(device)
#           optimizer = bert_adamw_optimizer(model, lr=args.lr, weight_decay=args.decay)
          
#     from spuco.evaluate import Evaluator
#     for i in range(args.epochs):
#         print('Epoch:',i)
#         jtt_train = UpSampleERM(
#               model=model,
#               num_epochs=1,
#               trainset=trainset,
#               batch_size=args.batch_size,
#               group_partition=disk_group_partition,
#               optimizer=optimizer,
#               device=device,
#               verbose=True)
#         jtt_train.train()  

#         evaluator = Evaluator(
#             testset=testset,
#             group_partition=testset.group_partition,
#             group_weights=trainset.group_weights,
#             batch_size=args.batch_size,
#             model=model,
#             device=device,
#             verbose=True)
#         evaluator.evaluate()
#         print(evaluator.worst_group_accuracy[1])
#         print(evaluator.average_accuracy)
#         with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
#                 f.write('Results at Epoch %0.0f: the worst acc is %0.4f and the average acc is %0.4f\n' % (i,evaluator.worst_group_accuracy[1],evaluator.average_accuracy))


    


