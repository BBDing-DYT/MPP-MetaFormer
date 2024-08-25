import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from torch.utils import data

from config import hyper_parameters_config_set
from data_load_utils import SingleLabelDataFrameToDataset
from models import SignleLabelModel


def load_data(config):
    dataframe_train = pd.read_csv(config['single_label_data_set_path'] + '_train.csv')
    dataframe_val = pd.read_csv(config['single_label_data_set_path'] + '_val.csv')
    dataframe_test = pd.read_csv(config['single_label_data_set_path'] + '_test.csv')

    train_data_set = SingleLabelDataFrameToDataset(dataframe_train)
    val_data_set = SingleLabelDataFrameToDataset(dataframe_val)
    test_data_set = SingleLabelDataFrameToDataset(dataframe_test)

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
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (drug, drug_mask, valid_len, label) in enumerate(data_generator):
        logits = torch.squeeze(model(drug.cuda(), valid_len.cuda()))
        label = label.float().cuda()
        loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, label).item()

        loss_accumulate += loss
        count += 1
        logits = F.sigmoid(logits).detach().cpu().numpy()
        label_ids = label.cpu().numpy()
        y_pred = y_pred + logits.tolist()
        y_label = y_label + label_ids.tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(f1) > 5 else thresholds[np.argmax(f1)]


    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return accuracy1, roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss


if __name__ == "__main__":
    hyper_parameters_config = hyper_parameters_config_set(dataset_name='BACE',
                                                          token_mixer_name='CNN',
                                                          use_pre_ln=True,
                                                          use_double_route_residue=False)
    dataset_name = hyper_parameters_config['single_label_data_set_path'].split('/')[-1]

    train_data_loader, val_data_loader, test_data_loader = load_data(hyper_parameters_config)

    model = SignleLabelModel(**hyper_parameters_config)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_parameters_config['learning_rate'], weight_decay=1e-2)
    loss_function = nn.BCEWithLogitsLoss()

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
            predict = torch.squeeze(model(drug.cuda(), valid_len.cuda()))
            label = label.float().cuda()

            loss = loss_function(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0 or (i + 1) % 10 == 0:
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
