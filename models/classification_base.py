import os
import numpy as np
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import h5py
import matplotlib.pyplot as plt
from plot import plot_confusion_matrix
from models.pkl import dump_json_file
# init logger
import logging
from models.logger import get_logger

logger = get_logger(min_level=logging.INFO)
#


class BaseClassificationNet(pl.LightningModule):

    def __init__(
        self,
        log_dir,
        batch_size,
        epochs,
        lr,
        dropout,
        chans,
        samples,
        num_classes=7,
        clip_length=4,
        clip_name="",
        loss_fn="ce",
        loss_weight=None,
        **kwargs,
    ):
        super(BaseClassificationNet, self).__init__()

        self.save_hyperparameters()

        self.log_dir = log_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.chans = chans
        self.samples = samples
        self.clip_name = clip_name
        self.num_classes = num_classes

        # loss functions
        assert loss_fn in ["ce"]
        if loss_fn == "ce":
            if loss_weight is None:
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                logger.info("use weighted cross entropy loss, {}".format(loss_weight))
                loss_weight = torch.tensor(loss_weight)
                self.loss_fn = nn.CrossEntropyLoss(weight=loss_weight)

        # softmax
        self.softmax = nn.Softmax(dim=-1)

        # metrics
        # for muti-class classification, overall accuracy is the same as weighted accuracy
        self.train_acc = torchmetrics.Accuracy(average="weighted", num_classes=num_classes)
        self.train_acc_balanced = torchmetrics.Accuracy(average="macro", num_classes=num_classes)
        self.train_f1_weighted = torchmetrics.F1Score(average="weighted", num_classes=num_classes)
        self.train_f1_balanced = torchmetrics.F1Score(average="macro", num_classes=num_classes)
        self.train_cohen_kappa = torchmetrics.CohenKappa(num_classes=num_classes)

        self.val_acc = torchmetrics.Accuracy(average="weighted", num_classes=num_classes)
        self.val_acc_balanced = torchmetrics.Accuracy(average="macro", num_classes=num_classes)
        self.val_f1_weighted = torchmetrics.F1Score(average="weighted", num_classes=num_classes)
        self.val_f1_balanced = torchmetrics.F1Score(average="macro", num_classes=num_classes)
        self.val_cohen_kappa = torchmetrics.CohenKappa(num_classes=num_classes)

        self.test_acc = torchmetrics.Accuracy(average="weighted", num_classes=num_classes)
        self.test_acc_balanced = torchmetrics.Accuracy(average="macro", num_classes=num_classes)
        self.test_f1_weighted = torchmetrics.F1Score(average="weighted", num_classes=num_classes)
        self.test_f1_balanced = torchmetrics.F1Score(average="macro", num_classes=num_classes)
        self.test_cohen_kappa = torchmetrics.CohenKappa(num_classes=num_classes)
        self.test_cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)

        # save to result
        self.test_f1_each = torchmetrics.F1Score(average=None, num_classes=num_classes)

    # models, let the child class to define
    #
    ###

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # use it to reshape the input data (tuple), let the child class to define
    def preprocess_input(self, x):
        print("preprocess_input is not implemented")
        pass

    # forward, let the child class to define
    def forward(self, *args):
        print("forward is not implemented")
        pass

    def _get_batch_data(self, batch):
        x, y = batch
        return x, y

    def training_step(self, batch, batch_idx):
        # x is tuple
        x, y = self._get_batch_data(batch)

        logits = self.forward(self.preprocess_input(x))

        loss = self.loss_fn(logits, y)

        y_prob = self.softmax(logits)

        # get true target label
        y = torch.argmax(y, dim=1)

        # update metrics
        self.train_acc(y_prob, y)
        self.train_acc_balanced(y_prob, y)
        self.train_f1_weighted(y_prob, y)
        self.train_f1_balanced(y_prob, y)
        self.train_cohen_kappa(y_prob, y)

        # calculate batch_size here because when I use pyg, the batch_size is batch_size*channels which is incorrect
        # use y.shape[0] instead
        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": self.train_acc,
                "train_acc_balanced": self.train_acc_balanced,
                "train_f1_weighted": self.train_f1_weighted,
                "train_f1_balanced": self.train_f1_balanced,
                "train_cohen_kappa": self.train_cohen_kappa,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.shape[0],
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._get_batch_data(batch)

        logits = self.forward(self.preprocess_input(x))

        loss = self.loss_fn(logits, y)
        y_prob = self.softmax(logits)

        # get true target label
        y = torch.argmax(y, dim=1)

        # update metrics
        self.val_acc(y_prob, y)
        self.val_acc_balanced(y_prob, y)
        self.val_f1_weighted(y_prob, y)
        self.val_f1_balanced(y_prob, y)
        self.val_cohen_kappa(y_prob, y)

        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": self.val_acc,
                "val_acc_balanced": self.val_acc_balanced,
                "val_f1_weighted": self.val_f1_weighted,
                "val_f1_balanced": self.val_f1_balanced,
                "val_cohen_kappa": self.val_cohen_kappa,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=y.shape[0],
        )

    def test_step(self, batch, batch_idx):
        x, y = self._get_batch_data(batch)

        logits = self.forward(self.preprocess_input(x))

        # loss = self.loss_fn(logits, y)
        y_prob = self.softmax(logits)

        # get true target label
        y = torch.argmax(y, dim=1)

        # update metrics
        self.test_acc(y_prob, y)
        self.test_acc_balanced(y_prob, y)
        self.test_f1_weighted(y_prob, y)
        self.test_f1_balanced(y_prob, y)
        self.test_cohen_kappa(y_prob, y)
        self.test_cm(y_prob, y)

        self.test_f1_each(y_prob, y)

        return y_prob.cpu().numpy()

    def test_epoch_end(self, test_step_outputs):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # compute the test metrics and clean
        acc = self.test_acc.compute()
        acc_balanced = self.test_acc_balanced.compute()
        cm = self.test_cm.compute()
        f1_weighted = self.test_f1_weighted.compute()
        f1_balanced = self.test_f1_balanced.compute()
        f1_each = self.test_f1_each.compute()
        cohen_kappa = self.test_cohen_kappa.compute()

        self.test_acc.reset()
        self.test_acc_balanced.reset()
        self.test_cm.reset()
        self.test_f1_weighted.reset()
        self.test_f1_balanced.reset()
        self.test_f1_each.reset()

        self.log_dict(
            {
                "test_acc_epoch": acc,
                "test_acc_balanced_epoch": acc_balanced,
                "test_f1_weighted_epoch": f1_weighted,
                "test_f1_balanced_epoch": f1_balanced,
                "test_cohen_kappa_epoch": cohen_kappa,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # use braindecode to plot confusion matrix
        from braindecode.visualization import plot_confusion_matrix
        fig = plot_confusion_matrix(
            cm.cpu().numpy(),
            class_names=["CF", "GN", "AB", "CT"],
            colormap=plt.cm.Blues,
            with_f1_score=True,
            rotate_precision=True,
        )
        fig.savefig(os.path.join(self.log_dir, "confusion_matrix.png"))

        result = {
            "acc": acc.cpu().numpy().tolist(),
            "acc_balanced": acc_balanced.cpu().numpy().tolist(),
            "f1_weighted": f1_weighted.cpu().numpy().tolist(),
            "f1_balanced": f1_balanced.cpu().numpy().tolist(),
            "f1_each": f1_each.cpu().numpy().tolist(),
            "cm": cm.cpu().numpy().tolist(),
            "cohen_kappa": cohen_kappa.cpu().numpy().tolist(),
        }

        import json
        print(json.dumps(result, indent=4))

        dump_json_file(os.path.join(self.log_dir, "results.json"), result)

        # save the model's prediction and true label to h5 file
        with h5py.File(os.path.join(self.log_dir, "model_preds_{}_class.h5".format(self.num_classes)), "w") as f:
            f.create_dataset("pred", data=np.concatenate(test_step_outputs))


