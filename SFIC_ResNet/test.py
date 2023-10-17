'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from network.data import TestDataset
from network.transform import Data_Transforms
from network.MainNet import MainNet
from network.plot_roc import plot_ROC
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    args = parse.parse_args()
    test_txt_path = args.test_txt_path
    batch_size = args.batch_size
    model_path = args.model_path
    num_classes = args.num_classes
	
    torch.backends.cudnn.benchmark=True
	
    # -----create train&val data----- #
    test_data = TestDataset(txt_path=test_txt_path, test_transform=Data_Transforms['test'])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # -----create model----- #
    model = MainNet(num_classes)
    model.load_state_dict(torch.load(model_path))
	
    if isinstance(model, nn.DataParallel):
        model = model.module
	
    model = model.cuda()
    model.eval()

    correct_test = 0.0
    total_test_samples = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img_rgb, labels_test = data

            img_rgb = img_rgb.cuda()
            labels_test = labels_test.cuda()
            
            # feed data
            pre_test = model(img_rgb)
            
            # prediction
            _, pred = torch.max(pre_test.data, 1)
            
            # the number of all testing sample
            total_test_samples += labels_test.size(0)
            
            # the correct number of prediction
            correct_test += (pred == labels_test).squeeze().sum().cpu().numpy()
            
            # compute ROC
            pre_test_abs = torch.nn.functional.softmax(pre_test, dim=1)
            pred_abs_temp = torch.zeros(pre_test_abs.size()[0])
            for m in range(pre_test_abs.size()[0]):
                pred_abs_temp[m] = pre_test_abs[m][1]

            label_test_list.extend(labels_test.detach().cpu().numpy())
            predict_test_list.extend(pred_abs_temp.detach().cpu().numpy())
            
        print("Testing Acc: {:.2%}".format(correct_test/total_test_samples))

    # ROC curve
    plot_ROC(label_test_list, predict_test_list)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--test_txt_path', '-tp', type=str, default='/home/yourpath/test.txt')
    parse.add_argument('--model_path', '-mp', type=str, default='./output/sfic-resnet/best.pkl')
    parse.add_argument('--num_classes', '-nc', type=int, default=2)
    
    label_test_list = []
    predict_test_list = []
    
    main()
