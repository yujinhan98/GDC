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
import time
import sys
from collections import defaultdict
from spuco.models import model_factory 
from spuco.utils import Trainer
from torch.optim import SGD
from spuco.robust_train import GroupDRO
from spuco.datasets import GroupLabeledDatasetWrapper
sys.path.append('/home/yujin/dm/disk')
from utils.load_data import load_cmnist, load_waterbirds, load_celebA, load_civilcomments
from utils.load_model import FineTuneResnet50,LeNet5,SpuriousNet,DISKNet,DISKNet_noy, bert_pretrained,bert_adamw_optimizer
from torchmetrics import Accuracy
import warnings
from spuco.robust_train import UpSampleERM, DownSampleERM, CustomSampleERM
sys.path.append('/home/yujin/dm/correct-n-contrast')
from network import get_net, get_optim, get_criterion, load_pretrained_model, save_checkpoint

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='The parameters of DISK')


parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs','-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='The initial learning rate.')
parser.add_argument('--decay','-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--momomentum', type=float, default=0.9, help='momomentum.')
parser.add_argument('--disk_epochs', type=int, default=15, help='Number of epochs to train disk.')
parser.add_argument('--disk_lr', type=float, default=1e-5, help='The initial learning rate of disk.')
parser.add_argument('--use_label', action='store_false', help='Use validation label or not')
parser.add_argument('--adjust', action='store_true', help='adjust eval or not')
parser.add_argument('--eval_weight', type=int, default=10, help='Loss weight of disk')
# parser.add_argument('--cluster_lr', type=float, default=1e-4, help='Description of cluster learning rate')
parser.add_argument('--hcs', type=float, default=None, help='High confidence sampling threshold')
# parser.add_argument('--mode', type=str, default='groupdro', choices = ['upweight','downweight','mixweight','groupdro'],help='Training Mode')
parser.add_argument('--upweight_factor', type=str, default='auto', help='10|20|100|auto')
parser.add_argument('--save_path' ,'-s', type=str, default='./snapshots_update/', help='Folder to save checkpoints.')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gpu', type=int, default=5) #7 6
# parser.add_argument('--r50', type=int, default=1)
parser.add_argument('--model_type', choices=['r50', 'bert', 'scratch-r50','lenet5'], default='r50')
parser.add_argument('--load',type=str, default=None, help='Folder to for extractor.')
parser.add_argument('--sload',type=str, default=None, help='Folder to for spurious extractor.')
parser.add_argument('--dataset', type=str, default='celebA', help='cmnist|waterbirds|celebA')
parser.add_argument('--es', type=str, default='eval', help='None|train|eval')
parser.add_argument('--mode', type=str, default='jtt', help='upweight|downweight|groupdro|jtt')
parser.add_argument('--lambdas', type=int, default=50, help='lambda')
parser.add_argument('--g1', type=int,default=None, help='Description of g1')
parser.add_argument('--g2', type=int,default=None, help='Description of g2')
parser.add_argument('--g3', type=int,default=None, help='Description of g3')
parser.add_argument('--g4', type=int,default=None, help='Description of g4')
args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))
# from resnet import ResNet_Model
# if args.mode == 'reweight':
#     save_info = 'reweight'
# elif args.score == 'dfr':
#     save_info = 'dfr'
# elif args.score == 'mixup':
#     save_info = 'mixup'

args.save = args.save_path + args.dataset + args.model_type+'_epoch'+str(args.epochs)+'_lr'+str(args.lr) + '_es' +str(args.es)+'_'+str(args.mode)#+'_g1'+str(args.g1)+'_adjust'
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
    train_loader_img = get_eval_loader("standard", train_dataset_img, batch_size=args.batch_size,pin_memory=True,num_workers=8)
    eval_loader_img = get_eval_loader("standard",eval_dataset_img, batch_size=args.batch_size,pin_memory=True,num_workers=8)
    test_loader_img = get_eval_loader("standard", test_dataset_img, batch_size=args.batch_size,pin_memory=True,num_workers=8)
elif args.dataset == 'celebA':
    train_dataset_img, eval_dataset_img,test_dataset_img = load_celebA()
    train_loader_img = get_eval_loader("standard", train_dataset_img, batch_size=args.batch_size,pin_memory=True,num_workers=8)
    eval_loader_img = get_eval_loader("standard",eval_dataset_img, batch_size=args.batch_size,pin_memory=True,num_workers=8)
    test_loader_img = get_eval_loader("standard", test_dataset_img, batch_size=args.batch_size,pin_memory=True,num_workers=8)
    if args.g1 is not None:
        def process_data(data):
            _, label, color_red = data
            return label.item(), color_red[0][0].item()
        print('Adjusting Eval...')
        processed_data = DataLoader(eval_dataset_img, batch_size=1, shuffle=False,pin_memory=True,num_workers=8)
        labels_eval_list = []
        cluster_labels_eval_list = []
        for data in tqdm(processed_data, total=len(processed_data), desc='Process Eval'):
            label, cluster_label = process_data(data)
            labels_eval_list.append(label)
            cluster_labels_eval_list.append(cluster_label)
        labels_eval_tensor = torch.tensor(labels_eval_list, dtype=torch.float)#.to(device)
        cluster_labels_eval_tensor = torch.tensor(cluster_labels_eval_list, dtype=torch.float)#.to(device)
        # processed_data = DataLoader(eval_loader_img.dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
        # labels_eval_tensor, cluster_labels_eval_tensor = zip(*map(process_data, processed_data))
        # labels_eval_tensor = torch.tensor(labels_eval_tensor, dtype=torch.float).to(device)
        # cluster_labels_eval_tensor = torch.tensor(cluster_labels_eval_tensor, dtype=torch.float).to(device)
        # print('Adjusting Eval...')
        # labels_eval_tensor = torch.tensor([label.item() for idx, (_, label, _) in enumerate(eval_loader_img.dataset)], dtype=torch.float).to(device)
        # cluster_labels_eval_tensor = torch.tensor([color_red[0].item() for idx, (_, _, color_red) in enumerate(eval_loader_img.dataset)], dtype=torch.float).to(device)
        print('Grouping...')
        same_predictions_indices_1 = torch.where((labels_eval_tensor == 1) & (labels_eval_tensor == cluster_labels_eval_tensor))[0]
        same_predictions_indices_0 = torch.where((labels_eval_tensor == 0) & (labels_eval_tensor == cluster_labels_eval_tensor))[0]
        different_predictions_indices_0 = torch.where((labels_eval_tensor == 0) & (labels_eval_tensor != cluster_labels_eval_tensor))[0]
        different_predictions_indices_1 = torch.where((labels_eval_tensor == 1) & (labels_eval_tensor != cluster_labels_eval_tensor))[0]
        print('Sampling ...')
        import torch.utils.data as data
        sample_same_predictions_indices_1 = random.choices(same_predictions_indices_1.tolist(), k=args.g4)
        sample_same_predictions_indices_0 = random.choices(same_predictions_indices_0.tolist(), k=args.g1)
        sample_different_predictions_indices_1 = random.choices(different_predictions_indices_1.tolist(), k=args.g3)
        sample_different_predictions_indices_0 = random.choices(different_predictions_indices_0.tolist(), k=args.g2)
        new_indices = sample_same_predictions_indices_1 + sample_same_predictions_indices_0 + sample_different_predictions_indices_1 + sample_different_predictions_indices_0
        print('Generating ...')
        subset_dataset = data.Subset(eval_dataset_img, new_indices)
        print('Loading ...')
        eval_loader_img = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True,num_workers=8)
        # labels_eval = np.array([label.item() for idx, (_, label, color_red) in enumerate(eval_loader_img.dataset)])
        # cluster_labels_eval = np.array([color_red[0].item() for idx, (_, label, color_red) in enumerate(eval_loader_img.dataset)])
        # same_predictions_indices_1 = np.where(np.logical_and(labels_eval == 1, labels_eval == cluster_labels_eval))[0]
        # same_predictions_indices_0 = np.where(np.logical_and(labels_eval == 0, labels_eval == cluster_labels_eval))[0]
        # different_predictions_indices_0 = np.where(np.logical_and(labels_eval == 0, labels_eval != cluster_labels_eval))[0]
        # different_predictions_indices_1 = np.where(np.logical_and(labels_eval == 1, labels_eval != cluster_labels_eval))[0]
        # sample_same_predictions_indices_1 = random.choices(list(same_predictions_indices_1), k=args.g4)
        # sample_same_predictions_indices_0 = random.choices(list(same_predictions_indices_0), k=args.g1)
        # sample_different_predictions_indices_1 = random.choices(list(different_predictions_indices_1), k=args.g3)
        # sample_different_predictions_indices_0 = random.choices(list(different_predictions_indices_0), k=args.g2)
        # new_indices = sample_same_predictions_indices_1 + sample_same_predictions_indices_0 + sample_different_predictions_indices_1 + sample_different_predictions_indices_0
        # subset_dataset = data.Subset(eval_loader_img.dataset, new_indices)
        # eval_loader_img = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)




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
    # net = load_pretrained_model('/home/yujin/dm/correct-n-contrast/model/celebA/celeba_erm_regularized.pt', args)
    # net = FineTuneResnet50().to(device)
    pretrained = False
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write(f"Load Pretrained Model {str(pretrained)}")
    net = FineTuneResnet50().to(device)
    if pretrained:
        model_path = '/home/yujin/dm/disk/snapshots/celebAr50_epoch5_lr0.0001_estrain/celebA_classifier_depoch20_dlr1e-05_labelTrue_hcsNone_s0_downweight_best_epoch.pt'
        net.load_state_dict(torch.load(model_path))

elif args.model_type == 'scratch-r50':
    net = FineTuneResnet50().to(device)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)
elif args.model_type == 'lenet5': #r50:
    net = LeNet5().to(device)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch: epoch,train_loss(%)\n')
    net = train_scratch(net, args.epochs)

       
# else:
#     # exteactor model load
#     net = ResNet_Model(name='resnet34', num_classes=num_classes)
if args.dataset == 'cmnist':
      worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == slabel]
elif args.dataset == 'waterbirds':
    worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label != slabel[0]]
elif args.dataset == 'celebA':
    worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == 1 and slabel[0] == 1]
    
worst_dataset = Subset(test_loader_img.dataset, worst_indices_test)
worst_loader = DataLoader(worst_dataset, batch_size= args.batch_size, shuffle=True)

def test_worst_(worst_loader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target,_ in worst_loader:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            pred = torch.squeeze(torch.round(torch.squeeze(output)))
            # pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    accuracy = correct / len(worst_loader.dataset)
    return accuracy


# /////////////// Training ///////////////
criterion_train = nn.BCELoss()#torch.nn.CrossEntropyLoss()
# optimizer_train = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.decay)
def train(train_loader_img,min_loader_img=None):
    net.train()
    correct = 0
    best_acc = 0
    best_epoch = 0
    total = 0
    optimizer_train = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    # if args.model_type == 'r50':
    #     for name, param in net.named_parameters():
    #         if "classifier" not in name:
    #             param.requires_grad = False
    #     optimizer_train = optim.SGD(net.classifier.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    # else:
    #     optimizer_train = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.decay,momentum=args.momomentum)
    start_time = time.time() 
    for epoch in range(args.epochs):
        running_loss = 0.0
        for (inputs, labels,_) in train_loader_img:
            optimizer_train.zero_grad()
            inputs, labels=inputs.to(device),labels.to(device)
            # print(inputs.shape)
            outputs = net(inputs)
            loss = criterion_train(outputs, labels.float().view(-1, 1))
            optimizer_train.zero_grad()
            loss.backward()
            optimizer_train.step()
            running_loss += loss.item()
            preds = torch.squeeze(torch.round(torch.squeeze(outputs)))
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        # scheduler.step()
        train_accuracy = correct / total
        test_acc = test_worst_(test_loader_img)
        test_worst_acc = test_worst_(worst_loader)
        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time 
        print('Epoch [%d/%d] | Time: %.2f seconds | Train Loss: %.4f | Train ACC: %.4f | Test ACC: %.4f | Test Worst ACC: %.4f' % (epoch + 1, args.epochs, epoch_time, running_loss / len(train_loader_img), train_accuracy, test_acc, test_worst_acc))
        start_time = time.time()

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
                    pred = torch.squeeze(torch.round(torch.squeeze(outputs)))
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
                                      f.write('%03d,%0.4f, %.4f, %.4f, %.4f, %.4f\n' % ((epoch + 1),running_loss/len(train_loader_img),train_accuracy,min_acc,test_acc,test_worst_acc))
    

def test():
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target,_ in test_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            # forward
            output = net(data)
            # accuracy
            # pred = output.data.max(1)[1]
            pred = torch.squeeze(torch.round(torch.squeeze(output)))
            correct += pred.eq(target.data).sum().item()
    test_accuracy = correct / len(test_loader_img.dataset)

    correct = 0
    with torch.no_grad():
        for data, target,_ in train_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            # pred = output.data.max(1)[1]
            pred = torch.squeeze(torch.round(torch.squeeze(output)))
            correct += pred.eq(target.data).sum().item()
    train_accuracy = correct / len(train_loader_img.dataset)

    correct = 0
    with torch.no_grad():
        for data, target,_ in eval_loader_img:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            # pred = output.data.max(1)[1]
            pred = torch.squeeze(torch.round(torch.squeeze(output)))
            correct += pred.eq(target.data).sum().item()
    eval_accuracy = correct / len(eval_loader_img.dataset)


    print('Train Accuracy: %.4f | Eval Accuracy: %.4f |Test Accuracy: %.4f ' % (train_accuracy,eval_accuracy, test_accuracy))
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
         f.write('Train Accuracy| Eval Accuracy|Test Accuracy: %0.4f,%0.4f,%0.4f\n' % (train_accuracy,eval_accuracy, test_accuracy))
    
def test_worst():
    # if args.dataset == 'cmnist':
    #     # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == slabel]
    #     # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == slabel]
    #     worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == slabel]
    # elif args.dataset == 'waterbirds':
    #     # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label != slabel[0]]
    #     # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label != slabel[0]]
    #     worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label != slabel[0]]
    # elif args.dataset == 'celebA':
    #     # worst_indices_train = [idx for idx, (_, label, slabel) in enumerate(train_loader_img.dataset) if label == 1 and slabel[0] == 1]
    #     # worst_indices_eval = [idx for idx, (_, label, slabel) in enumerate(eval_loader_img.dataset) if label == 1 and slabel[0] == 1]
    #     worst_indices_test = [idx for idx, (_, label, slabel) in enumerate(test_loader_img.dataset) if label == 1 and slabel[0] == 1]
    worst_dataset = Subset(test_loader_img.dataset, worst_indices_test)
    worst_loader = DataLoader(worst_dataset, batch_size= args.batch_size, shuffle=True)
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target,_ in worst_loader:
            data, target = data.to(device), target.to(device)#.cuda()
            output = net(data)
            pred = torch.squeeze(torch.round(torch.squeeze(output)))
            # pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    accuracy = correct / len(worst_loader.dataset)
    print('Test Worst Accuracy: %.4f ' % (accuracy))
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
         f.write('Test Worst Group Accuracy: %0.4f\n' % (accuracy))
    return accuracy

if args.model_type == 'scratch-r50':
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
           f.write('Training Scratch:\n')
    test()
    test_worst()
elif args.model_type == 'lenet5': #r50:
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
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
            if args.dataset == 'waterbirds' or args.dataset == 'celebA':
                  item2 = item[2][0].item()
            group = (item[1].item(), item2)  # Assuming the group information is at index 2
            group_counts[group] += 1
        # print(group_counts)
        group_weights = {group: count / total_samples for group, count in group_counts.items()}
        return group_weights
    def create_group_partition(self):
        group_partition = defaultdict(list)
        for idx, item in enumerate(self.data):
            group = (item[1].item(), item[2][0].item())  # Using a tuple of (label, group) as the key
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
# feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor[0].fc = torch.nn.Identity()


import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

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
    emb_dir = 'emb_data/'+'label'+str(args.use_label)+'_seed'+str(args.seed)+'_'+name+'_celebA_emb_data'
    os.makedirs(emb_dir, exist_ok=True)
    labels_list = []
    slabels_list = []
    for images, labels, slabels in tqdm(data_loader_img, desc="Extracting"):
        images, labels, slabels = images.to(device), labels.to(device), slabels.to(device)
        if args.dataset == 'waterbirds' or args.dataset == 'celebA':
            slabels = slabels[:, 0]
        outputs = feature_extractor(images).squeeze()
        # print('outputs.shape',outputs.shape)
        for i, emb in enumerate(outputs):
            torch.save(emb, os.path.join(emb_dir, f'emb_{len(labels_list) + i}.pt'))
        labels_list.extend(labels.tolist())
        slabels_list.extend(slabels.tolist())

    dataset = EmbDataset(emb_dir, labels_list, slabels_list)
    emb_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return emb_loader

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
            # print('inputs_train.shape)',inputs_train.shape)
            labels_train = labels_train.to(device)
            inputs_eval = inputs_eval.to(device)
            labels_eval = labels_eval.to(device)

            optimizer_spurious.zero_grad()
            optimizer_disk.zero_grad()

            outputs_train = spurious_net(inputs_train)
            outputs_eval = spurious_net(inputs_eval)

            # Prediction Loss
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
    spu_labels = []
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
            spu_labels.extend(slabel.cpu().numpy())
            correct += (pred == label).sum().item()#pred.eq(label.data).sum().item()
            scorrect += (pred == slabel).sum().item()#pred.eq(slabel.data).sum().item()

    accuracy = correct / len(data_loader_img.dataset)
    saccuracy = scorrect / len(data_loader_img.dataset)
    if saccuracy < 0.5 and args.mode == 'jtt':
          pred_labels = [1-value for value in pred_labels]
    saccuracy = max(saccuracy,1-saccuracy)
    return accuracy, saccuracy, pred_labels, true_labels,spu_labels




# class EmbDataset(Dataset):
#     def __init__(self, images, labels, slabels):
#         self.images = images
#         self.labels = labels
#         self.slabels = slabels

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx], self.slabels[idx]


# def embbeding(data_loader_img):
#     # fc_outputs = []
#     # labels_list = []
#     # slabels_list = []
#     dataset = EmbDataset([], [], [])
#     for images, labels, slabels in tqdm(data_loader_img, desc="Extracting"):
#         images, labels, slabels=images.to(device),labels.to(device),slabels.to(device)
#         if args.dataset=='waterbirds' or args.dataset=='celebA':
#             slabels = slabels[:, 0]
#         outputs = feature_extractor(images)
#         dataset.images.append(outputs)
#         dataset.labels.append(labels)
#         dataset.slabels.append(slabels)
#     emb_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
#     return emb_loader

# class EmbDataset(Dataset):
#     def __init__(self, data):
#         self.features = torch.tensor(data.iloc[:, :-2].values, dtype=torch.float32)
#         self.labels = torch.tensor(data.iloc[:, -2].values, dtype=torch.float32)
#         self.slabels = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, index):
#         feature = self.features[index]
#         label = self.labels[index]
#         slabel = self.slabels[index]
#         return feature, label, slabel

# def embbeding(data_loader_img,name):
#     fc_outputs = []
#     labels_list = []
#     slabels_list = []
#     print('Beginning Extracting\n')
#     for images, labels, slabels in tqdm(data_loader_img, desc='Processing images', unit='batch'):
#     # for images, labels, slabels in data_loader_img:
#         images, labels=images.to(device),labels.to(device)
#         if args.dataset=='waterbirds' or args.dataset=='celebA':
#             slabels = slabels[:, 0]
#         outputs = feature_extractor(images)
#         fc_output = outputs.view(outputs.size(0), -1).cpu().detach().numpy()
#         fc_outputs.append(fc_output)
#         labels_list.append(labels.cpu().numpy())
#         slabels_list.append(slabels.cpu().numpy())
    
#     all_fc_outputs = np.concatenate(fc_outputs, axis=0)
#     all_labels = np.concatenate(labels_list, axis=0)
#     all_slabels = np.concatenate(slabels_list, axis=0)
#     all_data = pd.concat([pd.DataFrame(all_fc_outputs),  pd.DataFrame(all_labels),  pd.DataFrame(all_slabels)], axis=1)
#     num_columns = all_data.shape[1]
#     column_names = ['F' + str(i) for i in range(1, num_columns-1)]
#     column_names.append('label')
#     column_names.append('slabel')
#     all_data.columns = column_names
#     emb_loader = DataLoader(EmbDataset(all_data), batch_size=args.batch_size, shuffle=False)
#     return emb_loader

# extract embedding

train_loader_emb =  embbeding(train_loader_img,'train')
eval_loader_emb =  embbeding(eval_loader_img,'eval')
test_loader_emb =  embbeding(test_loader_img,'test')


if args.adjust:
    dim = train_loader_emb.dataset[0][0].shape[0]
    spurious_net = SpuriousNet(data_dim = dim).to(device)
    state_dict = torch.load(args.sload, map_location=device)
    spurious_net.load_state_dict(state_dict)
    spurious_net.to(device)
    eval_accuracy, eval_saccuracy,eval_slabels,eval_labels = test_disk(eval_loader_emb)
    from collections import defaultdict
    indices_dict = defaultdict(list)
    for idx, (label, slabel) in enumerate(zip(eval_labels, eval_slabels)):
        key = (label, slabel)
        indices_dict[key].append(idx)
    for key, indices in indices_dict.items():
          print(f"Eval Indices for key {key}: {len(indices)}")
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs) + '_dlr' + str(args.disk_lr) + '_label' + str(args.use_label) + '_hcs' + str(args.hcs) + '_s' + str(args.seed) + '_' + args.mode + str(args.upweight_factor) + '_training_results.csv'), 'a') as f:
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
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs) + '_dlr' + str(args.disk_lr) + '_label' + str(args.use_label) + '_hcs' + str(args.hcs) + '_s' + str(args.seed) + '_' + args.mode + str(args.upweight_factor) + '_training_results.csv'), 'a') as f:
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





# /////////////// Training DISK ///////////////

dim = train_loader_emb.dataset[0][0].shape[0]
# dim = test_loader_emb.dataset[0][0].shape[0]




spurious_net = SpuriousNet(data_dim = dim).to(device)
if args.use_label:
    disk_net = DISKNet().to(device)
else:
    disk_net = DISKNet_noy(data_dim = dim).to(device)

optimizer_spurious =  optim.SGD(spurious_net.parameters(), lr=args.disk_lr, momentum=0.9)
optimizer_disk = optim.SGD(disk_net.parameters(),  lr=args.disk_lr, momentum=0.9)

# def train_disk(train_loader, eval_loader,start_avg_et=1.0,unbiased_loss=False):
#     # all_mi = []
#     # CE = []
#     # MINE = []
#     avg_et = start_avg_et
#     for epoch in range(args.disk_epochs):
#         running_loss = 0.0
#         kl_loss = 0.0
#         pre_loss = 0.0
#         for (inputs_train, labels_train, _), (inputs_eval, labels_eval, _) in zip(train_loader, eval_loader):
#             inputs_train = inputs_train.to(device)
#             # print('inputs_train.shape)',inputs_train.shape)
#             labels_train = labels_train.to(device)
#             inputs_eval = inputs_eval.to(device)
#             labels_eval = labels_eval.to(device)

#             optimizer_spurious.zero_grad()
#             optimizer_disk.zero_grad()

#             outputs_train = spurious_net(inputs_train)
#             outputs_eval = spurious_net(inputs_eval)

#             # Prediction Loss
#             loss_cluster_train = nn.BCELoss()(outputs_train, labels_train.float().view(-1, 1))

#             # MI Loss
#             if args.use_label:
#                 KL_loss,avg_et = disk_net(labels_train.float().view(-1, 1),labels_eval.float().view(-1, 1),outputs_train,outputs_eval,avg_et,unbiased_loss=unbiased_loss)
#             else:
#                 KL_loss,avg_et = disk_net(inputs_train, inputs_eval, outputs_train, outputs_eval,avg_et,unbiased_loss=unbiased_loss)
            
#             # Final Loss
#             loss = loss_cluster_train - args.eval_weight * KL_loss

#             loss.backward()
#             optimizer_spurious.step()
#             optimizer_disk.step()

#             kl_loss = KL_loss.item()
#             pre_loss = loss_cluster_train.item()
#             running_loss = loss.item()

#         print('Epoch [%d/%d]| Train Loss: %.4f | KL Loss: %.4f | Prediction Loss: %.4f' % (epoch+1, args.disk_epochs, running_loss, kl_loss, pre_loss))

#         torch.save(spurious_net.state_dict(),
#                os.path.join(args.save, args.dataset + '_spurious' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
#                             '_' + args.mode+ '_epoch_' + str(epoch) + '.pt'))
#         # Let us not waste space and delete the previous model
#         prev_path = os.path.join(args.save, args.dataset + '_spurious' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
#                             '_' + args.mode+ '_epoch_' + str(epoch - 1) + '.pt')
#         if os.path.exists(prev_path): os.remove(prev_path)
#         with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
#                                       f.write('%03d,%0.4f,%0.4f,%0.4f\n' % ((epoch + 1),running_loss,kl_loss,pre_loss))


# def test_disk(data_loader_img,hcs=args.hcs):
#     spurious_net.eval()
#     correct = 0
#     scorrect = 0
#     pred_labels = []
#     # pred_probs = []
#     true_labels = []
#     with torch.no_grad():
#         for data_emb, label,slabel in data_loader_img:
#             data_emb, label,slabel = data_emb.to(device), label.to(device), slabel.to(device)#.cuda()
#             output = spurious_net(data_emb)
#             # print(output)
#             # pred = output.data.max(1)[1]
#             pred = torch.squeeze(torch.round(torch.squeeze(output)))
#             if hcs is not None:
#                 for i in range(len(pred)):
#                     if pred[i] != label[i] and hcs < output[i] < 1-hcs:
#                            pred[i] = label[i]
#             pred_labels.extend(pred.cpu().numpy())
#             # pred_probs.extend(torch.squeeze(output).cpu().numpy())
#             true_labels.extend(label.cpu().numpy())
#             correct += (pred == label).sum().item()#pred.eq(label.data).sum().item()
#             scorrect += (pred == slabel).sum().item()#pred.eq(slabel.data).sum().item()

#     accuracy = correct / len(data_loader_img.dataset)
#     saccuracy = scorrect / len(data_loader_img.dataset)
#     saccuracy = max(saccuracy,1-saccuracy)
#     return accuracy, saccuracy, pred_labels, true_labels




with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    for key, value in state.items():
        f.write('%s:%s\n' % (key, value))
with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    f.write('epoch,train_loss,KL_loss,pre_error(%)\n')

print('Beginning Training DISK\n')
train_disk(train_loader_emb, eval_loader_emb)
# if args.sload is None:
#       train_disk(train_loader_emb, eval_loader_emb)
# else:
#       state_dict = torch.load(args.sload, map_location=device)
#       spurious_net.load_state_dict(state_dict)
#       spurious_net.to(device)


print('Beginning Testing DISK\n')
with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('DISK: train_accuracy,eval_accuracy,test_accuracy\n')
train_accuracy, train_saccuracy, train_slabels, train_labels, train_spu_labels= test_disk(train_loader_emb)
eval_accuracy, eval_saccuracy,eval_slabels,eval_labels, eval_spu_labels  = test_disk(eval_loader_emb)
test_accuracy, test_saccuracy,_,_,_ = test_disk(test_loader_emb)
print('Train Accuracy: %.4f | Eval Accuracy: %.4f | Test Accuracy: %.4f' % (train_accuracy,eval_accuracy,test_accuracy))
print('Train SAccuracy: %.4f | Eval SAccuracy: %.4f | Test SAccuracy: %.4f' % (train_saccuracy,eval_saccuracy,test_saccuracy))
with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%0.4f,%0.4f,%0.4f\n' % (train_accuracy,eval_accuracy,test_accuracy))

with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('DISK: train_saccuracy,eval_saccuracy,test_saccuracy\n')

with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                                      f.write('%0.4f,%0.4f,%0.4f\n' % (train_saccuracy,eval_saccuracy,test_saccuracy))


# /////////////// Retraining///////////////

from collections import defaultdict

indices_dict = defaultdict(list)
group_len = []
for idx, (label, slabel) in enumerate(zip(train_labels, train_slabels)):
    key = (label, slabel)
    indices_dict[key].append(idx)

if args.mode == 'groupdro' or args.mode == 'jtt':
    disk_group_partition = dict(indices_dict)
for key, indices in indices_dict.items():
    print(f"Indices for key {key}: {len(indices)}")
    group_len.append(len(indices))
group_len = sorted(group_len)
with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
    f.write('Predicted Train Group Info:\n')
    for key, indices in indices_dict.items():
        f.write(f"Key {key} has {len(indices)} samples before balancing\n")

if args.es == 'train':
    min_group = min(indices_dict, key=lambda k: len(indices_dict[k]))

if args.mode ==  'upweight':
       max_samples = max(len(indices) for indices in indices_dict.values())
       balanced_indices = []
       for key, indices in indices_dict.items():
              ratio = int(max_samples / len(indices))
              repeated_indices = [idx for idx in indices for _ in range(ratio)]
              indices_dict[key] = repeated_indices
        
       for key, indices in indices_dict.items():
        print(f"Key {key} has {len(indices)} samples after balancing")

       with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
        f.write('Group Info:\n')
        for key, indices in indices_dict.items():
            f.write(f"Indices for key {key}: {len(indices)}\n")
            f.write(f"Key {key} has {len(indices)} samples after balancing\n")

elif args.mode == 'downweight':
    min_samples = min(len(indices) for indices in indices_dict.values())
    balanced_indices = []
    if round((min_samples/len(train_loader_img.dataset))*100) < 5:
        thres = len(train_loader_img.dataset)*0.05/4
        min_samples = next((indices for indices in group_len if indices > thres), min_samples)
        # print('thres:',thres,group_len)
        # print('min_samples:',min_samples)
        # min_samples = next((len(indices) for indices in indices_dict.values() if len(indices) > thres), min_samples)
        for key, indices in indices_dict.items():
            if len(indices) > min_samples:
                sampled_indices = random.sample(indices, min_samples)  # Downsample
                indices_dict[key] = sampled_indices
            else:
                ratio = int(min_samples / len(indices))
                repeated_indices = [idx for idx in indices for _ in range(ratio)]
                indices_dict[key] = repeated_indices
            print(f"Key {key} has {len(indices_dict[key])} samples after balancing")
    else:
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
elif args.mode == 'mixweight':
    min_samples = sorted([len(indices) for indices in indices_dict.values()])[::-1][1]  # Get the second smallest sample size
    balanced_indices = []
    for key, indices in indices_dict.items():
        if len(indices) > min_samples:
              sampled_indices = random.sample(indices, min_samples)  # Downsample
              indices_dict[key] = sampled_indices
        else:
              ratio = int(min_samples / len(indices))
              repeated_indices = [idx for idx in indices for _ in range(ratio)]
              indices_dict[key] = repeated_indices
            #   oversampled_indices = random.choices(indices, k=min_samples - len(indices))  # Upsample
            #   indices_dict[key].extend(oversampled_indices)
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
    balanced_indices = []
    for key, indices in indices_dict.items():
        if key in keys:
            ratio = args.lambdas
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




if args.es is not None:
    if args.es == 'eval':
        eval_indices_dict = defaultdict(list)
        for idx, (label, slabel) in enumerate(zip(eval_labels, eval_spu_labels)):
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
        min_loader_img = data.DataLoader(eval_mingroup, batch_size=args.batch_size, shuffle=True,pin_memory=True,num_workers=8)
    elif args.es == 'train':
        # min_group = min(indices_dict, key=lambda k: len(indices_dict[k]))
        min_train_samples = len(indices_dict[min_group])
        min_group_indices = indices_dict[min_group]
        print(f"The train group with the minimum number of samples is {min_group} with {min_train_samples} samples.")
        with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write(f"The train group with the minimum number of samples is {min_group} with {min_train_samples} samples.")
        train_mingroup = data.Subset(train_loader_img.dataset, min_group_indices)
        min_loader_img = data.DataLoader(train_mingroup, batch_size=args.batch_size, shuffle=True,pin_memory=True,num_workers=8)
if args.es == None:
    min_loader_img = None


if args.mode in  ['upweight','downweight','mixweight','jtt']:
    original_dataset = train_loader_img.dataset
    sampled_indices = []
    for indices in indices_dict.values():
        sampled_indices.extend(indices)
    sampled_dataset = data.Subset(original_dataset, sampled_indices)
    sampled_train_loader_img = data.DataLoader(sampled_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True,num_workers=8)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
          f.write('\nepoch,train_loss,train ACC, min ACC, test ACC,test worst Acc\n')
    train(sampled_train_loader_img,min_loader_img)
    with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
            f.write('Final Results: \n') 
# test()
# test_worst()

# if args.es is not None:
#     state_dict = torch.load(os.path.join(args.save, args.dataset + '_classifier' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr) +'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) +
#                             '_' + args.mode+ '_best_epoch' + '.pt'), map_location=device)
#     if args.model_type == 'r50': 
#         net = FineTuneResnet50().to(device)
#     elif args.model_type == 'lenet5': 
#         net = LeNet5().to(device)
#     elif args.model_type == 'bert': 
#         net = bert_pretrained(output_dim=2).to(device)
#     net.load_state_dict(state_dict)
#     net.to(device)
#     net.eval()    
#     with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
#             f.write('Best Results: \n')   
#     test()
#     test_worst()

elif args.mode ==  'groupdro':
    from spuco.evaluate import Evaluator
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

    for i in range(args.epochs):
          print('Epoch:',i)
          group_dro = GroupDRO(
            model=model,
            num_epochs=1,
            trainset=group_labeled_trainset,
            batch_size=args.batch_size,
            optimizer=optimizer,
            device=device,
            verbose=True
            )
          group_dro.train()
          evaluator = Evaluator(
            testset=testset,
            group_partition=testset.group_partition,
            group_weights=trainset.group_weights,
            batch_size=args.batch_size,
            model=model,
            device=device,
            verbose=True)
          evaluator.evaluate()
          print(evaluator.worst_group_accuracy[1])
          print(evaluator.average_accuracy)
          with open(os.path.join(args.save, args.dataset + '_DISK' + '_depoch' + str(args.disk_epochs)+'_dlr'+str(args.disk_lr )+'_label'+ str(args.use_label) +'_hcs'+str(args.hcs)+ '_s' + str(args.seed) + '_' + args.mode +str(args.upweight_factor)+ '_training_results.csv'), 'a') as f:
                f.write('Results at Epoch %0.0f: the worst acc is %0.4f and the average acc is %0.4f\n' % (i,evaluator.worst_group_accuracy[1],evaluator.average_accuracy))

    # group_dro = GroupDRO(
    #         model=model,
    #         num_epochs=args.epochs,
    #         trainset=group_labeled_trainset,
    #         batch_size=args.batch_size,
    #         optimizer=optimizer,
    #         device=device,
    #         verbose=True
    #         )
    # group_dro.train()
    # from spuco.evaluate import Evaluator
    # evaluator = Evaluator(
    #         testset=testset,
    #         group_partition=testset.group_partition,
    #         group_weights=trainset.group_weights,
    #         batch_size=args.batch_size,
    #         model=model,
    #         device=device,
    #         verbose=True)
    # evaluator.evaluate()
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


    


                  

