import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch
import warnings
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models.model import Classification_FBCNetWithAttn_MultiSpatialConv as My_Model
from models.model import FBCNet
from configuration import args
from my_trainer import train, val,test
from plot import plot_loss,plot_acc
import csv
import torch.nn.functional as F
import torchmetrics
#ignore warning
import logging
from models.logger import get_logger
from torchsummary import summary

logger = get_logger(min_level=logging.INFO)
warnings.filterwarnings('ignore')
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = 'cpu'
def loading(dir_path):
    data_path = dir_path+'processed.npy'
    label_path = dir_path+'labels.npy'
    x = np.load(data_path,)
    y = np.load(label_path,)

    X = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return X,y

def calc_ce_class_weight(labels_count, mu_ratio):
    total = 0
    for key in labels_count:
        total += labels_count[key]

    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes

    logger.info("calculate class weight, total clips: {}, num classes: {}, factor: {}".format(
        total, num_classes, factor))

    factor_mu = [factor * mu for mu in mu_ratio]

    logger.info("munally set mu: {}".format(factor_mu))

    for key in range(num_classes):
        score = math.log(factor_mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * factor_mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    logger.info("class weight for cross entropy: {}".format(class_weight))

    return torch.Tensor(class_weight)

if __name__ == "__main__":
    #:data_eeg, label_eeg, data_hbr, label_hbr
    train_path = 'Data/mult_view/train_'
    train_x, train_label = loading(train_path)
    print(train_x.shape, train_label.shape)


    test_path = 'Data/mult_view/dev_'
    test_x, test_label = loading(test_path)
    print(test_x.shape, test_label.shape)

    val_path = 'Data/mult_view/eval_'
    val_x, val_label = loading(val_path)
    print(val_x.shape, val_label.shape)
    print(val_label)


    print("loading dataset done")

    # 模型
    model = My_Model(
        log_dir="",
        batch_size=64,
        epochs=200,
        num_classes=7,
        lr=0.0001,
        dropout=0.5,
        chans=19,
        samples=512).to(device)
    # model = FBCNet(
    #     log_dir="",
    #     batch_size=64,
    #     epochs=200,
    #     num_classes=7,
    #     lr=0.0001,
    #     dropout=0.5,
    #     chans=19,
    #     samples=512).to(device)



    print("loading model done...")
    LABELS_COUNT = {
        0: 23097,
        1: 11524,
        2: 460,
        3: 6748,
        4: 95,
        5: 106,
        6: 477
    }
    # eva: 3374 2133 0 833 44 9 148
    # dev: 4358 6875 0 15 28 36 682
    # train: 23097 11524 460 6748 95 106 477

    mu_ratio = [1, 1, 1, 1, 1, 1, 1]
    loss_wigtht = calc_ce_class_weight(LABELS_COUNT, mu_ratio)
    loss_wigtht = loss_wigtht.to(device)
    # print(loss_wigtht)
    s_optimizer = optim.Adam(params=model.parameters(),lr=args.lr)#,
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(s_optimizer, gamma=0.9999)

    loss_fn = nn.CrossEntropyLoss() #nn.CrossEntropyLoss() #weight=loss_wigtht

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(s_optimizer, gamma=0.95)
    # 学习率每隔10轮变为原来的0.5
    # StepLR：用于调整学习率，一般情况下会设置随着epoch的增大而逐渐减小学习率从而达到更好的训练效果
    # optimizer （Optimizer）：更改学习率的优化器
    # step_size（int）：每训练step_size个epoch，更新一次参数
    # gamma（float）：更新lr的乘法因子
    #    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # 打印模型摘要信息
    # summary(model, (9, 19, 512))

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    # 用来记录最优的正确率
    best_acc = 0.0
    iteration = 0
    train_times = 0.0
    num_classes = 7
    for i in range(args.epochs):
        print("=========epoch {}=========".format(i + 1))
        train_times = 0.0
        epoch = i
        num_classes = 7
        # train_x, train_label = train_x.to(device), train_label.to(device)
        train_loss, train_acc,labels = train(model, train_x, train_label,loss_fn, s_optimizer,num_classes)
        # visualize(feature.cpu().data.numpy(), labels.cpu().data.numpy(), epoch)
        # 验证模型
        # val_x, vali_label = val_x.to(device), val_label.to(device)
        val_loss, val_acc = val(model,val_x, val_label, loss_fn,num_classes)

        loss_train.append(train_loss)
        acc_train.append(train_acc)
        loss_val.append(val_loss)
        acc_val.append(val_acc)

        # 保存最好的模型权重
        if val_acc > best_acc:
            folder = 'save_model/'  # 设置模型存储路径
            # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
            if not os.path.exists(folder):
                # os.mkdir() 方法用于以数字权限模式创建目录
                os.mkdir('save_model/')
            best_acc = val_acc
            print(f"save best model，第{i + 1}轮", i + 1)
            # torch.save(state, dir)：保存模型等相关参数，dir表示保存文件的路径+保存文件名
            # model.state_dict()：返回的是一个OrderedDict，存储了网络结构的名字和对应的参数
            torch.save(model.state_dict(), 'save_model/model.pth')

    folder = 'images/'  # 设置模型存储路径
    # path.exists：判断括号里的文件是否存在的意思，括号内可以是文件路径，存在为True
    if not os.path.exists(folder):
        # os.mkdir() 方法用于以数字权限模式创建目录
        os.mkdir('images/')
    plot_loss(loss_train, loss_val,'loss')
    plot_acc(acc_train, acc_val,'acc')
    print('Training done!')
    print('Testing star...')

    # 模型实例化，将模型转到device
    model = My_Model(
        log_dir="",
        batch_size=64,
        epochs=200,
        lr=0.0001,
        dropout=0.5,
        chans=19,
        samples=512).to(device)
    # model = FBCNet(
    #     log_dir="",
    #     batch_size=64,
    #     epochs=200,
    #     num_classes=7,
    #     lr=0.0001,
    #     dropout=0.5,
    #     chans=19,
    #     samples=512).to(device)

    model.load_state_dict(torch.load(r'save_model/model.pth'))

    model.eval()
    x = test_x
    y = test_label
    val_loss, val_acc,weighted_f1,balanced_f1,cohen_kappa,cm = test(model, x, y, loss_fn, num_classes)



