'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import torch
import numpy as np
import random
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torchvision


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# save tensors as pics.
def save_pic(tensor, save_path):
    for i in range(tensor.size(0)):
        torchvision.utils.save_image(tensor[i,:,:,:], save_path+'/{}.png'.format(i))


def cal_metrics(y_true_all, y_pred_all):

    fprs, tprs, ths = roc_curve(
        y_true_all, y_pred_all, pos_label=1, drop_intermediate=False)
    
    auc_value = auc(fprs, tprs)
    eer = brentq(lambda x : 1. - x - interp1d(fprs, tprs)(x), 0., 1.)

    ind = 0
    for fpr in fprs:
        if fpr > 1e-2:
            break
        ind += 1
    TPR_2 = tprs[ind-1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-3:
            break
        ind += 1
    TPR_3 = tprs[ind-1]

    ind = 0
    for fpr in fprs:
        if fpr > 1e-4:
            break
        ind += 1
    TPR_4 = tprs[ind-1]

    ap = average_precision_score(y_true_all, y_pred_all)
    return ap, auc_value, eer, TPR_2, TPR_3, TPR_4
    
# plot ROC, compute AUC and EER
def plot_ROC(y, y_p):
    fpr, tpr, thresholds = roc_curve(y, y_p)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    #print('EER:', eer)
    #print('Thresholds:', thresh)
    ax3 = plt.subplot()
    ax3.set_title("Receiver Operating Characteristic", verticalalignment='center')
    plt.plot(fpr, tpr, 'b', label='AUC=%0.4f ' % roc_auc)
    #print('AUC:', roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('True Positive Rate(TPR)')
    plt.xlabel('False Positive Rate(FPR)')
    plt.savefig('./output/roc.png', bbox_inches='tight')
    #plt.show()
    plt.pause(3)
    return eer, roc_auc

