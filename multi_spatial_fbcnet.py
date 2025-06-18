### update sys path
import sys

PRED_WORKDIR = "/home/cage/workspace/sz-pred/my-kaggle-pred"
TUH_WORKDIR = "/home/cage/workspace/sz-pred/tuh-detection"

sys.path.append(PRED_WORKDIR)
sys.path.append(TUH_WORKDIR)
###

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipes.fbcnet.common import Conv2dWithConstraint, LinearWithConstraint, VarLayer, LogVarLayer, swish, LogLayer

from tuh.models.detection_base import BaseDetectionNet
from tuh.models.classification_base import BaseClassificationNet

# init logger
import logging
from src.utils.logger import get_logger

logger = get_logger(min_level=logging.WARN)
#


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # avg and max
        channel = channel * 2
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(channel // reduction, channel // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # mean (batch size, m*bands, stride) -> (batch size,stride, m*bands)
        # AvgPool
        x_avg = x.mean(dim=3)
        # MaxPool
        x_max, _ = x.max(dim=3, keepdim=False)
        x = torch.concat((x_avg, x_max), dim=1)
        x = x.permute(0, 2, 1)
        #  (batch size,stride, m*bands) -> (batch size,stride, m*bands // 16) -> (batch size,stride, m*bands )
        y = self.fc(x)
        return y


def LastBlock(inF, outF, doWeightNorm=False, *args, **kwargs):
    return nn.Sequential(LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs))


class MultiSpatialFBCNet(BaseClassificationNet):

    def __init__(self,
                 log_dir,
                 batch_size,
                 epochs,
                 lr,
                 dropout,
                 chans,
                 samples,
                 num_classes=4,
                 clip_length=4,
                 clip_name="",
                 loss_fn="ce",
                 loss_weight=None,
                 **kwargs):
        super().__init__(log_dir, batch_size, epochs, lr, dropout, chans, samples, num_classes, clip_length, clip_name,
                         loss_fn, loss_weight, **kwargs)

        # for FBCNet:
        self.m_for_each_band = kwargs.get("m_for_each_band", 8)
        self.m = kwargs.get("m", 32)
        assert self.m_for_each_band * 4 == self.m

        self.n_band = kwargs.get("n_band", 5)
        self.n_stride = kwargs.get("n_stride", 4)
        self.n_selayer_reduction = kwargs.get("n_selayer_reduction", 16)
        self.conv2d_type = kwargs.get("conv2d_type", "conv2d_with_constraint")

        if self.conv2d_type == "conv2d":
            Conv2d = nn.Conv2d
        elif self.conv2d_type == "conv2d_with_constraint":
            Conv2d = Conv2dWithConstraint

        # abulation study
        self.is_band_attn = kwargs.get("is_band_attn", True)
        # 是否使用rnn
        self.is_rnn = kwargs.get("is_rnn", True)
        # 这里是使用lstm还是bilstm
        self.is_bilstm = kwargs.get("is_bilstm", True)
        # 这里是bilstm是否使用attn
        self.is_bilstm_attn = kwargs.get("is_bilstm_attn", True)

        self.n_lstm_hidden = kwargs.get("n_lstm_hidden", 32)
        self.n_lstm_layers = kwargs.get("n_lstm_layer", 2)
        # self.bidirectional = kwargs.get("bidirectional", True)
        self.lstm_dropout = kwargs.get("lstm_dropout", 0.5)

        self.spatial_conv_1 = Conv2d(
            self.n_band,
            self.m_for_each_band * self.n_band,
            kernel_size=(3, 1),
            padding="same",
            groups=self.n_band,
            max_norm=2,
        )
        self.spatial_conv_2 = Conv2d(
            self.n_band,
            self.m_for_each_band * self.n_band,
            kernel_size=(5, 1),
            padding="same",
            groups=self.n_band,
            max_norm=2,
        )
        self.spatial_conv_3 = Conv2d(
            self.n_band,
            self.m_for_each_band * self.n_band,
            kernel_size=(3, 1),
            dilation=(2, 1),
            padding="same",
            groups=self.n_band,
            max_norm=2,
        )
        self.spatial_conv_4 = Conv2d(
            self.n_band,
            self.m_for_each_band * self.n_band,
            kernel_size=(5, 1),
            dilation=(2, 1),
            padding="same",
            groups=self.n_band,
            max_norm=2,
        )
        self.batch_norm_1 = nn.BatchNorm2d(self.m_for_each_band * self.n_band)
        self.batch_norm_2 = nn.BatchNorm2d(self.m_for_each_band * self.n_band)
        self.batch_norm_3 = nn.BatchNorm2d(self.m_for_each_band * self.n_band)
        self.batch_norm_4 = nn.BatchNorm2d(self.m_for_each_band * self.n_band)

        self.dropout = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.swish = swish()

        # 将通道维度进行压缩
        self.scb = nn.Sequential(
            Conv2d(
                self.m * self.n_band,
                self.m * self.n_band,
                (self.chans, 1),
                groups=self.n_band,
                max_norm=2,
                doWeightNorm=True,
                padding=0,
            ),
            nn.BatchNorm2d(self.m * self.n_band),
            swish(),
        )

        self.temporal_layer = LogVarLayer(dim=3)

        # 是否使用band attn
        if self.is_band_attn:
            self.se_layer = SELayer(channel=self.m * self.n_band, reduction=self.n_selayer_reduction)

        # 是否使用rnn
        if self.is_rnn:
            # 是否具有bi-lstm attn
            if self.is_bilstm_attn:
                self.tanh = nn.Tanh()
                self.w = nn.Parameter(torch.zeros(self.n_lstm_hidden * 2))

            self.rnn = nn.LSTM(input_size=self.m * self.n_band,
                               hidden_size=self.n_lstm_hidden,
                               num_layers=self.n_lstm_layers,
                               batch_first=True,
                               bidirectional=self.is_bilstm,
                               dropout=self.lstm_dropout)

        # 无rnn
        if not self.is_rnn:
            self.fc = LastBlock(self.m * self.n_band * self.n_stride, num_classes, doWeightNorm=True)
        # 有rnn
        else:
            # bi-lstm
            if self.is_bilstm:
                self.fc = LastBlock(self.n_lstm_hidden * 2, num_classes, doWeightNorm=True)
            # lstm
            else:
                self.fc = LastBlock(self.n_lstm_hidden, num_classes, doWeightNorm=True)

    def preprocess_input(self, x):
        return x.type(torch.cuda.FloatTensor)

    def forward(self, *args):
        x = args[0]

        x_1 = self.batch_norm_1(self.spatial_conv_1(x))
        x_2 = self.batch_norm_2(self.spatial_conv_2(x))
        x_3 = self.batch_norm_3(self.spatial_conv_3(x))
        x_4 = self.batch_norm_4(self.spatial_conv_4(x))

        x_1 = x_1.reshape(x_1.shape[0], self.n_band, -1, x_1.shape[2], x_1.shape[3])
        x_2 = x_2.reshape(x_2.shape[0], self.n_band, -1, x_2.shape[2], x_2.shape[3])
        x_3 = x_3.reshape(x_3.shape[0], self.n_band, -1, x_3.shape[2], x_3.shape[3])
        x_4 = x_4.reshape(x_4.shape[0], self.n_band, -1, x_4.shape[2], x_4.shape[3])

        x = torch.cat([x_1, x_2, x_3, x_4], dim=2)

        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

        x = self.scb(x)

        x = x.reshape([*x.shape[0:2], self.n_stride, -1])

        temporal_var = self.temporal_layer(x)

        # band attn
        if self.is_band_attn:
            band_attentions = self.se_layer(x)
            band_attentions = band_attentions.permute(0, 2, 1).unsqueeze(3)
            attn_temporal_var = torch.mul(band_attentions, temporal_var)
        else:
            attn_temporal_var = temporal_var

        # 是否使用rnn
        if not self.is_rnn:
            var = torch.flatten(attn_temporal_var, start_dim=1)
            x = self.dropout(var)
            x = self.fc(x)
        else:
            # reshape to [batch_size, seq_len, feature]
            attn_temporal_var = attn_temporal_var.squeeze(3)
            var = attn_temporal_var.permute(0, 2, 1)

            H, _ = self.rnn(var)
            if self.is_bilstm_attn:
                M = self.tanh(H)
                alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
                # with attn
                H = H * alpha
            # bilstm是每个时序的输出进行相加
            if self.is_bilstm:
                rnn_out = torch.sum(H, dim=1)
            # lstm是最后一个时序的输出
            else:
                rnn_out = H[:, -1, :]

            x = self.fc(rnn_out)

        return x


class MultiSpatialFBCNetTransformer(MultiSpatialFBCNet):

    def __init__(self,
                 log_dir,
                 batch_size,
                 epochs,
                 lr,
                 dropout,
                 chans,
                 samples,
                 num_classes=4,
                 clip_length=4,
                 clip_name="",
                 loss_fn="ce",
                 loss_weight=None,
                 **kwargs):
        super().__init__(log_dir, batch_size, epochs, lr, dropout, chans, samples, num_classes, clip_length, clip_name,
                         loss_fn, loss_weight, **kwargs)

        # add transformer encoder
        self.transformer_nheads = kwargs.get("transformer_nheads", 4)
        self.transformer_layer_dropout = kwargs.get("transformer_layer_dropout", 0.5)
        self.transformer_dim_feedforward = kwargs.get("transformer_dim_feedforward", 64)
        self.transformer_num_layers = kwargs.get("transformer_layers", 2)
        # replace lstm with transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.m * self.n_band,
                                                   nhead=self.transformer_nheads,
                                                   dim_feedforward=self.transformer_dim_feedforward,
                                                   dropout=0.5,
                                                   batch_first=True)
        self.rnn = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_num_layers)

        self.fc = LastBlock(self.n_stride * self.m * self.n_band, num_classes, doWeightNorm=True)

    def preprocess_input(self, x):
        return x.type(torch.cuda.FloatTensor)

    def forward(self, *args):
        x = args[0]

        # spatial conv
        x_1 = self.relu(self.batch_norm_1(self.spatial_conv_1(x)))
        x_2 = self.relu(self.batch_norm_2(self.spatial_conv_2(x)))
        x_3 = self.relu(self.batch_norm_3(self.spatial_conv_3(x)))
        x_4 = self.relu(self.batch_norm_4(self.spatial_conv_4(x)))

        x_1 = x_1.reshape(x_1.shape[0], self.n_band, -1, x_1.shape[2], x_1.shape[3])
        x_2 = x_2.reshape(x_2.shape[0], self.n_band, -1, x_2.shape[2], x_2.shape[3])
        x_3 = x_3.reshape(x_3.shape[0], self.n_band, -1, x_3.shape[2], x_3.shape[3])
        x_4 = x_4.reshape(x_4.shape[0], self.n_band, -1, x_4.shape[2], x_4.shape[3])

        x = torch.cat([x_1, x_2, x_3, x_4], dim=2)

        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

        x = self.scb(x)

        x = x.reshape([*x.shape[0:2], self.n_stride, -1])

        # temporal var
        temporal_var = self.temporal_layer(x)

        # band attn
        if self.is_band_attn:
            band_attentions = self.se_layer(x)
            band_attentions = band_attentions.permute(0, 2, 1).unsqueeze(3)
            attn_temporal_var = torch.mul(band_attentions, temporal_var)
        else:
            attn_temporal_var = temporal_var

        # 是否使用rnn
        if not self.is_rnn:
            var = torch.flatten(attn_temporal_var, start_dim=1)
            x = self.dropout(var)
            x = self.fc(x)
        else:
            # reshape to [batch_size, seq_len, feature]
            attn_temporal_var = attn_temporal_var.squeeze(3)
            var = attn_temporal_var.permute(0, 2, 1)

            # replace lstm with transformer encoder
            rnn_out = self.rnn(var)
            rnn_out = rnn_out.reshape(rnn_out.shape[0], -1)

            x = self.fc(rnn_out)

        return x


if __name__ == '__main__':
    transformer = MultiSpatialFBCNetTransformer(log_dir="",
                                                batch_size=32,
                                                epochs=100,
                                                lr=0.001,
                                                dropout=0.5,
                                                chans=19,
                                                samples=512).cuda()
    input = torch.randn(32, 5, 19, 512).cuda()
    output = transformer(input)
    print(transformer)
    print(output.shape)
