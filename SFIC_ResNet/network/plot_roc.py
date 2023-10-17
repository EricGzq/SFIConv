'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
from sklearn.metrics import roc_curve,auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# plot ROC, compute AUC and EER
def plot_ROC(y, y_p):
    fpr, tpr, thresholds = roc_curve(y, y_p)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print('EER:', eer)
    print('Thresholds:', thresh)
    ax3 = plt.subplot()
    ax3.set_title("Receiver Operating Characteristic", verticalalignment='center')
    plt.plot(fpr, tpr, 'b', label='AUC=%0.4f ' % roc_auc)
    print('AUC:', roc_auc)
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
