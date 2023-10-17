'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import os
import torch
import time
import platform
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from network.MainNet import MainNet
from network.data import *
from network.transform import Data_Transforms
from datetime import datetime

from network.log_record import *
from network.pipeline import *
from network.utils import setup_seed, cal_metrics, plot_ROC

print('-'*20)
print("PyTorch version:{}".format(torch.__version__))
print("Python version:{}".format(platform.python_version()))
print("cudnn version:{}".format(torch.backends.cudnn.version()))
print("GPU name:{}".format(torch.cuda.get_device_name(0)))
print("GPU number:{}".format(torch.cuda.device_count()))
print('-'*20)

def main():
    args = parse.parse_args()

    name = args.name
    train_txt_path = args.train_txt_path
    valid_txt_path = args.valid_txt_path
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    num_classes = args.num_classes

    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    log_path = os.path.join(output_path)
    
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    print('Training datetime: ', time_str)
    
    torch.backends.cudnn.benchmark = True

    # -----create train&val data----- #
    train_data = SingleInputDataset(txt_path=train_txt_path, train_transform=Data_Transforms['train'])
    valid_data = SingleInputDataset(txt_path=valid_txt_path, valid_transform=Data_Transforms['val'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # -----create the model----- #
    setup_seed(args.seed)
    model = MainNet(num_classes)
    
    # -----calculate FLOPs and Params----- #
    flops, params = cal_params_ptflops(model, (3, 256, 256))
    print('{:<30}  {:<8}'.format('Computational complexity (FLOPs): ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters (Params): ', params))   

    if continue_train:
        model.load_state_dict(torch.load(model_path))
    

    # -----define the loss----- #
    criterion = nn.CrossEntropyLoss()

    # -----define the optimizer----- #
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # -----define the learning rate scheduler----- #
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # -----multiple gpus for training----- #
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    
    # -----------------------------------------define the train & val----------------------------------------- #
    best_acc = 0.0
    best_auc = 0.0
    time_open = time.time()

    for epoch in range(epoches):
        total_train_samples = 0.0
        correct_tra = 0.0
        sum_loss_tra = 0.0

        print('\nEpoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)

        # -----Training----- #
        for i, data in enumerate(train_loader):
            img_train, labels_train = data
            
            # input images & labels
            img_train = img_train.cuda()
            labels_train = labels_train.cuda()  # labels_train.size(0) = batchsize = 64

            optimizer.zero_grad()
            model=model.train()
            
            # feed data to model
            pre_tra = model(img_train)

            # the average loss of a batch
            loss_tra = criterion(pre_tra, labels_train)
            sum_loss_tra += loss_tra.item() * labels_train.size(0)
            
            # prediction
            _, pred = torch.max(pre_tra.data, 1)

            loss_tra.backward()
            optimizer.step()

            # the correct number of prediction
            correct_tra += (pred == labels_train).squeeze().sum().cpu().numpy()

            # the number of all training samples
            total_train_samples += labels_train.size(0)

            # training information is printed every 100 iterations. NOTE: i starts from 0.
            if i % 100 == 99:
                print("Training: Epoch[{:0>1}/{:0>1}] Iteration[{:0>1}/{:0>1}] Loss:{:.2f} Acc:{:.2%}".format(epoch + 1, epoches, i + 1, len(train_loader), sum_loss_tra/total_train_samples, correct_tra/total_train_samples))

        # -----Validating----- #
        if epoch % 1 == 0:
            sum_loss_val = 0.0
            correct_val = 0.0
            total_valid_samples = 0.0

            model.eval()
            
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    img_valid, labels_valid = data

                    img_valid = img_valid.cuda()
                    labels_valid = labels_valid.cuda()
                                
                    pre_val = model(img_valid)

                    # compute the loss
                    loss_val = criterion(pre_val, labels_valid)
                    sum_loss_val += loss_val.item() * labels_valid.size(0)

                    # prediction
                    _, pred = torch.max(pre_val.data, 1)

                    # the number of all validating sample
                    total_valid_samples += labels_valid.size(0)

                    # the correct number of prediction
                    correct_val += (pred == labels_valid).squeeze().sum().cpu().numpy()

                    # prepare for ROC
                    pre_val_abs = torch.nn.functional.softmax(pre_val, dim=1)
                    pred_abs_temp = torch.zeros(pre_val_abs.size()[0])
                    for m in range(pre_val_abs.size()[0]):
                        pred_abs_temp[m] = pre_val_abs[m][1]

                    label_val_list.extend(labels_valid.detach().cpu().numpy())
                    predict_val_list.extend(pred_abs_temp.detach().cpu().numpy())
      
                # acc
                epoch_acc = correct_val / total_valid_samples
                
                # auc
                if num_classes == 2:
                    ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4 = cal_metrics(label_val_list, predict_val_list)
                    
                    # save the results
                    save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch, log_path)
                    
                    if epoch_auc > best_auc:
                        best_auc = epoch_auc
                    
                else:
                    ap_score = 0
                    epoch_auc = 0
                    epoch_eer = 0
                    TPR_2 = 0
                    TPR_3 = 0
                    TPR_4 = 0
                    # save the results
                    save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch, log_path)
                
                print("Validating: Epoch[{:0>1}/{:0>1}] Acc:{:.2%} Auc:{:.2%}".format(epoch + 1, epoches, epoch_acc, epoch_auc))
                
                # select the best accuracy and save the best pretrained model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if multiple_gpus:
                        best_model_wts = model.module.state_dict()
                        torch.save(best_model_wts, os.path.join(output_path, "best.pkl"))
                    else:
                        best_model_wts = model.state_dict()
                        torch.save(best_model_wts, os.path.join(output_path, "best.pkl"))
                  
            # update learning rate 
            scheduler.step()    # for other strategy

        # -----save the pretrained model----- #
        if epoch+1 == epoches:
            if multiple_gpus:
                torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch+1) + '_epoches_' + model_name))
            else:
                torch.save(model.state_dict(), os.path.join(output_path, str(epoch+1) + '_epoches_' + model_name))
    
    # ----------------------------------------------------end----------------------------------------------------#
    
    # -----print the results----- #
    print('-'*20)        
    print('Best_accuracy:', best_acc)
    print('Best_AUC:', best_auc)
    # print time
    time_end = time.time() - time_open
    print('All time: ', time_end)

    # -----save final results----- #
    if num_classes == 2:
        plot_ROC(label_val_list, predict_val_list)
        save_final_results(flops, params, time_end, best_acc, best_auc, log_path)
    else:
        best_auc = 0
        save_final_results(flops, params, time_end, best_acc, best_auc, log_path)



if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='sfic-resnet')
    parse.add_argument('--train_txt_path', '-tp', type=str, default = '/home/yourpath/train.txt')
    parse.add_argument('--valid_txt_path', '-vp', type=str, default = '/home/yourpath/val.txt')
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default=20)
    parse.add_argument('--model_name', '-mn', type=str, default='sfic-resnet.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./output/SFI-resnet26/best.pkl')
    parse.add_argument('--num_classes', '-nc', type=int, default=2)
    parse.add_argument('--seed', default=7, type=int)
    
    multiple_gpus = True
    gpus = [0,1]

    label_val_list = []
    predict_val_list = []
    
    main()
