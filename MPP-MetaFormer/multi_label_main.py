import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch.utils import data

from config import hyper_parameters_config_set
from data_load_utils import MultiLabelDataFrameToDataSet
from models import Tox21MultiLabelModel


def load_data(config):
    dataframe_train = pd.read_csv(config['multi_label_data_set_path'] + '_train.csv')
    dataframe_val = pd.read_csv(config['multi_label_data_set_path'] + '_val.csv')
    dataframe_test = pd.read_csv(config['multi_label_data_set_path'] + '_test.csv')

    train_data_set = MultiLabelDataFrameToDataSet(dataframe_train)
    val_data_set = MultiLabelDataFrameToDataSet(dataframe_val)
    test_data_set = MultiLabelDataFrameToDataSet(dataframe_test)

    data_loader_params_train = {'batch_size': config['batch_size'],
                                'shuffle': True,
                                'num_workers': config['workers'],
                                'drop_last': config['drop_last_train']}
    data_loader_params_val_and_test = {'batch_size': config['batch_size'],
                                       'shuffle': False,
                                       'num_workers': config['workers'],
                                       'drop_last': config['drop_last_val_and_test']}
    train_data_loader = data.DataLoader(train_data_set, **data_loader_params_train)
    val_data_loader = data.DataLoader(val_data_set, **data_loader_params_val_and_test)
    test_data_loader = data.DataLoader(test_data_set, **data_loader_params_val_and_test)

    return train_data_loader, val_data_loader, test_data_loader


def test(data_generator, model):
    class_num = len(data_generator.dataset.df.columns) - 1
    y_pred = [[] for _ in range(class_num)]
    y_label = [[] for _ in range(class_num)]
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (drug, drug_mask, valid_len, label) in enumerate(data_generator):
        loss_mask = (label != -1).float().cuda()
        label = label.float().cuda()

        logits = model(drug.cuda(), valid_len.cuda())
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        loss = loss_fct(logits, label)
        loss = loss * loss_mask
        loss = (loss.sum() / loss_mask.sum()).item()

        loss_accumulate += loss
        count += 1

        logits = F.sigmoid(logits)

        mask = (label != -1)

        label = label.transpose(0, 1)
        pred = logits.transpose(0, 1)
        mask = mask.transpose(0, 1)

        for i in range(class_num):
            current_class_pred = pred[i][mask[i]].detach().cpu().numpy()
            current_class_label = label[i][mask[i]].cpu().numpy()
            y_pred[i].extend(current_class_pred.tolist())
            y_label[i].extend(current_class_label.tolist())

    loss = loss_accumulate / count

    actual_class_num = 0
    auroc_accumulate = 0
    aurpc_accumulate = 0
    recall_accumulate = 0
    precision_accumulate = 0
    f1_score_accumulate = 0
    accuracy_accumulate = 0
    sensitivity_accumulate = 0
    specificity_accumulate = 0
    y_pred_s = []
    for i in range(class_num):
        fpr, tpr, thresholds = roc_curve(y_label[i], y_pred[i])

        precision = tpr / (tpr + fpr)
        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
        thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(f1) > 5 else thresholds[np.argmax(f1)]

        y_pred_s.append([1 if j else 0 for j in (y_pred[i] >= thred_optim)])

        auroc_accumulate += auc(fpr, tpr)
        aurpc_accumulate += average_precision_score(y_label[i], y_pred[i])

        cm1 = confusion_matrix(y_label[i], y_pred_s[i])
        recall_accumulate += recall_score(y_label[i], y_pred_s[i])
        precision_accumulate += precision_score(y_label[i], y_pred_s[i])
        f1_score_accumulate += f1_score(y_label[i], y_pred_s[i])

        total = sum(sum(cm1))
        accuracy_accumulate += (cm1[0, 0] + cm1[1, 1]) / total
        sensitivity_accumulate += cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity_accumulate += cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

        actual_class_num = actual_class_num + 1

    print("AUROC:" + str(auroc_accumulate / actual_class_num))
    print("AUPRC: " + str(aurpc_accumulate / actual_class_num))
    print('Recall : ', str(recall_accumulate / actual_class_num))
    print('Precision : ', str(precision_accumulate / actual_class_num))
    print('Accuracy : ', str(accuracy_accumulate / actual_class_num))
    print('Sensitivity : ', str(sensitivity_accumulate / actual_class_num))
    print('Specificity : ', str(specificity_accumulate / actual_class_num))

    return accuracy_accumulate / actual_class_num, auroc_accumulate / actual_class_num, aurpc_accumulate / actual_class_num, f1_score_accumulate / actual_class_num, loss


if __name__ == "__main__":
    hyper_parameters_config = hyper_parameters_config_set(dataset_name='Tox21',
                                                          token_mixer_name='Self-Attention',
                                                          use_pre_ln=True,
                                                          use_double_route_residue=False)
    dataset_name = hyper_parameters_config['multi_label_data_set_path'].split('/')[-1]

    train_data_loader, val_data_loader, test_data_loader = load_data(hyper_parameters_config)

    model = Tox21MultiLabelModel(**hyper_parameters_config)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_parameters_config['learning_rate'], weight_decay=1e-2)
    loss_function = nn.BCEWithLogitsLoss(reduction='none')

    use_pre_ln = hyper_parameters_config['use_pre_ln']
    use_double_route_residue = hyper_parameters_config['use_double_route_residue']
    val_weights_file = ("weights/" + "useDoubleRouteResidue_" + str(use_double_route_residue) + "_" +
                        "usePreLN_" + str(use_pre_ln) + "_" +
                        dataset_name + "_" + hyper_parameters_config['token_mixer_name'] + "_val.pth")
    test_weights_file = ("weights/" + "useDoubleRouteResidue_" + str(use_double_route_residue) + "_" +
                         "usePreLN_" + str(use_pre_ln) + "_" +
                         dataset_name + "_" + hyper_parameters_config['token_mixer_name'] + "_test.pth")

    if os.path.exists(val_weights_file):
        checkpoint = torch.load(val_weights_file)
        best_val_auroc = checkpoint['val_auroc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("loss: " + str(checkpoint['val_loss']))
        print("AUROC:" + str(checkpoint['val_auroc']))
        print("AUPRC:" + str(checkpoint['val_auprc']))
    else:
        best_val_auroc = 0

    if os.path.exists(test_weights_file):
        checkpoint = torch.load(test_weights_file)
        best_test_auroc = checkpoint['test_auroc']
    else:
        best_test_auroc = 0

    last_best_val_auroc = best_val_auroc
    last_best_test_auroc = best_test_auroc

    best_model = copy.deepcopy(model)
    for epoch in range(hyper_parameters_config['train_epoch']):
        model.train()
        for i, (drug, drug_mask, valid_len, label) in enumerate(train_data_loader):
            predict = model(drug.cuda(), valid_len.cuda())
            loss_mask = (label != -1).float().cuda()
            label = label.float().cuda()

            loss = loss_function(predict, label)
            loss = loss * loss_mask
            loss = loss.sum() / loss_mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0 or (i + 1) % 5 == 0:
                print('Training at Epoch ' + str(epoch + 1) + ' iteration ' + str(i + 1) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        found_better_val_model = False
        with torch.set_grad_enabled(False):
            val_acc, val_auc, val_auprc, val_f1, val_loss = test(val_data_loader, model)
            if val_auc > best_val_auroc:
                found_better_val_model = True
                best_val_auroc = val_auc
                best_model = copy.deepcopy(model)
                state = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_auroc": val_auc,
                    "val_auprc": val_auprc,
                    "val_f1": val_f1
                }
                torch.save(state, val_weights_file)
            print('Validation at Epoch ' + str(epoch + 1) + ' , AUROC: ' + str(val_auc) + ' , AUPRC: ' + str(
                val_auprc) + ' , ACC: ' + str(val_acc) + ' , F1: ' + str(val_f1) + ' , loss: ' + str(val_loss))

        with torch.set_grad_enabled(False):
            if found_better_val_model:
                test_acc, test_auc, test_auprc, test_f1, test_loss = test(test_data_loader, best_model)
                if test_auc > best_test_auroc:
                    best_test_auroc = test_auc
                    state = {
                        "state_dict": best_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "test_auroc": test_auc,
                        "test_auprc": test_auprc,
                        "test_f1": test_f1
                    }
                    torch.save(state, test_weights_file)
                print('Test at Epoch ' + str(epoch + 1) + ' , AUROC: ' + str(test_auc) + ' , AUPRC: ' + str(
                    test_auprc) + ' , ACC: ' + str(test_acc) + ' , F1: ' + str(test_f1) + ' , loss: ' + str(test_loss))
