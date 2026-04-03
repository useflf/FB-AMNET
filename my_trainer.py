# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#
# import torch
# import torchmetrics
# import warnings
# from configuration import args
# from plot import nextBatch
# import numpy as np
# import torch.nn.functional as F
# import csv
# import logging
# from models.logger import get_logger
#
# logger = get_logger(min_level=logging.INFO)
# warnings.filterwarnings('ignore')
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# def loading(dir_path):
#     data_path = dir_path + 'processed.npy'
#     label_path = dir_path + 'labels.npy'
#     x = np.load(data_path)
#     y = np.load(label_path)
#
#     X = torch.from_numpy(x)
#     y = torch.from_numpy(y)
#
#     return X, y
#
#
# def train(s_net, train_data, train_label, loss_fn, s_optimizer, num_classes):
#     """
#     注意：这里假设传进来的 train_label 已经在 main.py 里 remap 成 0..(num_classes-1) 了
#     """
#     s_net.train()
#     loss, current, n = 0.0, 0.0, 0
#     label_loader = []
#
#     for batch in nextBatch(train_data, train_label, args.batch_size):
#         x = batch[0]
#         y = batch[1]
#
#         if args.iscuda:
#             x, y = x.to(device), y.to(device)
#
#         # 保证 y 是 [B]，且为 long
#         y = y.view(-1).long()
#
#         y_hat = s_net(x)  # [B, C]
#
#         # one-hot: [B, C]
#         label = F.one_hot(y, num_classes).float()
#
#         l = loss_fn(y_hat.float(), label)
#
#         _, preds = torch.max(y_hat, dim=1)
#         cur_acc = torch.sum(y == preds) / y_hat.shape[0]
#
#         s_optimizer.zero_grad()
#         l.backward()
#         s_optimizer.step()
#
#         label_loader.append(y.detach())
#         loss += l.item()
#         current += cur_acc.item()
#         n += 1
#
#     train_loss = loss / n
#     train_acc = current / n
#     labels = torch.cat(label_loader, dim=0)
#
#     print('train_loss=' + str(train_loss), 'train_acc=' + str(train_acc))
#     return train_loss, train_acc, labels
#
#
# def val(s_net, vali_eeg, vali_label, loss_fn, num_classes):
#     loss, current, n = 0.0, 0.0, 0
#     s_net.eval()
#
#     with torch.no_grad():
#         for batch in nextBatch(vali_eeg, vali_label, args.batch_size):
#             x = batch[0]
#             y = batch[1]
#
#             if args.iscuda:
#                 x, y = x.to(device), y.to(device)
#
#             y = y.view(-1).long()
#             output = s_net(x)
#
#             label = F.one_hot(y, num_classes).float()
#             s_loss = loss_fn(output.float(), label)
#
#             _, pred = torch.max(output, dim=1)
#             cur_acc = torch.sum(y == pred) / output.shape[0]
#
#             loss += s_loss.item()
#             current += cur_acc.item()
#             n += 1
#
#     val_loss = loss / n
#     val_acc = current / n
#     print('val_loss=' + str(val_loss), 'val_acc=' + str(val_acc))
#     return val_loss, val_acc
#
#
# def test(s_net, vali_eeg, vali_label, loss_fn, num_classes):
#     loss, current, n = 0.0, 0.0, 0
#     cm = []
#     label_loader = []
#     preds_all = []
#
#     s_net.eval()
#     with torch.no_grad():
#         for batch in nextBatch(vali_eeg, vali_label, args.batch_size):
#             x = batch[0]
#             y = batch[1]
#
#             if args.iscuda:
#                 x, y = x.to(device), y.to(device)
#
#             y = y.view(-1).long()
#             output = s_net(x)
#
#             label = F.one_hot(y, num_classes).float()
#             s_loss = loss_fn(output.float(), label)
#
#             _, preds = torch.max(output, dim=1)
#             cur_acc = torch.sum(y == preds) / output.shape[0]
#
#             loss += s_loss.item()
#             current += cur_acc.item()
#             label_loader.append(y.detach())
#             preds_all.append(preds.detach())
#             n += 1
#
#     labels = torch.cat(label_loader, dim=0)
#     pre_labels = torch.cat(preds_all, dim=0)
#
#     weighted_f1 = torchmetrics.functional.f1_score(
#         labels, pre_labels, task="multiclass", average="weighted", num_classes=num_classes
#     )
#     balanced_f1 = torchmetrics.functional.f1_score(
#         labels, pre_labels, task="multiclass", average="macro", num_classes=num_classes
#     )
#     cohen_kappa = torchmetrics.functional.cohen_kappa(
#         labels, pre_labels, task="multiclass", num_classes=num_classes
#     )
#     cm = torchmetrics.functional.confusion_matrix(
#         labels, pre_labels, task="multiclass", num_classes=num_classes
#     )
#
#     val_loss = loss / n
#     val_acc = current / n
#
#     print('test_loss=' + str(val_loss), 'test_acc=' + str(val_acc))
#     print('w_f1=' + str(weighted_f1), 'balanced_f1=' + str(balanced_f1), 'cohen_kappa=' + str(cohen_kappa))
#
#     with open('test.csv', 'a+', newline='', encoding='utf-8') as file:
#         swriter = csv.writer(file)
#         swriter.writerow([
#             'test_loss', str(val_loss), 'test_acc', str(val_acc),
#             'weighted_f1', str(weighted_f1), 'balanced_f1', str(balanced_f1),
#             'cohen_kappa', str(cohen_kappa), 'cm', str(cm)
#         ])
#
#     return val_loss, val_acc, weighted_f1, balanced_f1, cohen_kappa, cm

# my_train.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torchmetrics
import warnings
from configuration import args
from plot import nextBatch
import numpy as np
import csv
import logging
from models.logger import get_logger

logger = get_logger(min_level=logging.INFO)
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loading(dir_path):
    data_path = dir_path + 'processed.npy'
    label_path = dir_path + 'labels.npy'
    x = np.load(data_path)
    y = np.load(label_path)
    return torch.from_numpy(x), torch.from_numpy(y)


def train(s_net, train_data, train_label, loss_fn, s_optimizer, num_classes):
    s_net.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    label_loader = []

    for xb, yb in nextBatch(train_data, train_label, args.batch_size):
        if args.iscuda:
            xb, yb = xb.to(device), yb.to(device)

        yb = yb.view(-1).long()                 # [B]
        logits = s_net(xb)                      # [B, C]  (logits)

        l = loss_fn(logits, yb)                 # CrossEntropyLoss(logits, class_index)

        preds = torch.argmax(logits, dim=1)
        cur_acc = (preds == yb).float().mean()

        s_optimizer.zero_grad()
        l.backward()
        s_optimizer.step()

        label_loader.append(yb.detach())
        loss_sum += l.item()
        acc_sum += cur_acc.item()
        n += 1

    train_loss = loss_sum / max(n, 1)
    train_acc = acc_sum / max(n, 1)
    labels = torch.cat(label_loader, dim=0)

    print('train_loss=' + str(train_loss), 'train_acc=' + str(train_acc))
    return train_loss, train_acc, labels


def val(s_net, vali_eeg, vali_label, loss_fn, num_classes):
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    s_net.eval()

    with torch.no_grad():
        for xb, yb in nextBatch(vali_eeg, vali_label, args.batch_size):
            if args.iscuda:
                xb, yb = xb.to(device), yb.to(device)

            yb = yb.view(-1).long()
            logits = s_net(xb)

            l = loss_fn(logits, yb)
            preds = torch.argmax(logits, dim=1)
            cur_acc = (preds == yb).float().mean()

            loss_sum += l.item()
            acc_sum += cur_acc.item()
            n += 1

    val_loss = loss_sum / max(n, 1)
    val_acc = acc_sum / max(n, 1)
    print('val_loss=' + str(val_loss), 'val_acc=' + str(val_acc))
    return val_loss, val_acc


def test(s_net, vali_eeg, vali_label, loss_fn, num_classes):
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    label_loader, preds_all = [], []

    s_net.eval()
    with torch.no_grad():
        for xb, yb in nextBatch(vali_eeg, vali_label, args.batch_size):
            if args.iscuda:
                xb, yb = xb.to(device), yb.to(device)

            yb = yb.view(-1).long()
            logits = s_net(xb)

            l = loss_fn(logits, yb)
            preds = torch.argmax(logits, dim=1)
            cur_acc = (preds == yb).float().mean()

            loss_sum += l.item()
            acc_sum += cur_acc.item()
            label_loader.append(yb.detach())
            preds_all.append(preds.detach())
            n += 1

    labels = torch.cat(label_loader, dim=0)
    pre_labels = torch.cat(preds_all, dim=0)

    weighted_f1 = torchmetrics.functional.f1_score(
        labels, pre_labels, task="multiclass", average="weighted", num_classes=num_classes
    )
    macro_f1 = torchmetrics.functional.f1_score(
        labels, pre_labels, task="multiclass", average="macro", num_classes=num_classes
    )
    cohen_kappa = torchmetrics.functional.cohen_kappa(
        labels, pre_labels, task="multiclass", num_classes=num_classes
    )
    cm = torchmetrics.functional.confusion_matrix(
        labels, pre_labels, task="multiclass", num_classes=num_classes
    )

    test_loss = loss_sum / max(n, 1)
    test_acc = acc_sum / max(n, 1)

    print('test_loss=' + str(test_loss), 'test_acc=' + str(test_acc))
    print('w_f1=' + str(weighted_f1), 'macro_f1=' + str(macro_f1), 'cohen_kappa=' + str(cohen_kappa))

    with open('test.csv', 'a+', newline='', encoding='utf-8') as f:
        swriter = csv.writer(f)
        swriter.writerow([
            'test_loss', str(test_loss), 'test_acc', str(test_acc),
            'weighted_f1', str(weighted_f1), 'macro_f1', str(macro_f1),
            'cohen_kappa', str(cohen_kappa), 'cm', str(cm)
        ])

    return test_loss, test_acc, weighted_f1, macro_f1, cohen_kappa, cm


if __name__ == "__main__":
    pass
