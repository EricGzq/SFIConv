'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
import os

def save_weights(w_rgb, w_res, epoch_num, log_dir):
    filename = open(os.path.join(log_dir, 'Weights.txt'), 'a')   # 'a': append, not overwrite;  'w': overwrite previous data
    fusion_data_save = 'Epoch_' + str(epoch_num+1) + ':' + str(w_rgb) + ' ' + str(w_res) + '\n'
    filename.write(fusion_data_save)


def save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch_num, log_dir):
    filename = open(os.path.join(log_dir, 'final_results.txt'), 'a')
    fusion_data_save = 'Epoch_' + str(epoch_num+1) + ':' + ' ' + 'acc:%.4f'%epoch_acc + ' ' + 'ap:%.4f'%ap_score + ' ' + 'auc:%.4f'%epoch_auc + ' ' + 'eer:%.4f'%epoch_eer + ' ' + 'TPR_2:%.4f'%TPR_2 + ' ' + 'TPR_3:%.4f'%TPR_3 + ' ' + 'TPR_4:%.4f'%TPR_4 + ' ' + '\n'
    filename.write(fusion_data_save)

def save_final_results(flops, params_count, time, best_acc, best_auc, log_dir):
    filename = open(os.path.join(log_dir, 'final_results.txt'), 'a')
    fusion_data_save = '-'*10 + '\n' + 'FLOPs: ' + str(flops) + '\n' + 'Params: ' + str(params_count) + '\n' + 'All time: ' + str(time) + '\n' + 'Best accuracy:%.4f'%best_acc + '\n' + 'Best AUC:%.4f'%best_auc + '\n' + '\n'
    filename.write(fusion_data_save)

