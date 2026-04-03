
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import Conv2dWithConstraint, LinearWithConstraint, VarLayer, LogVarLayer, swish
from models.classification_base import BaseClassificationNet

#init logger
import logging
from models.logger import get_logger

logger = get_logger(min_level=logging.WARN)



class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # avg and max
        channel = channel * 2
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
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


def SCB(m, n_chan, n_band, do_weight_norm, **kwargs):
    return nn.Sequential(
        Conv2dWithConstraint(n_band,
                             m * n_band, (n_chan, 1),
                             groups=n_band,
                             max_norm=2,
                             doWeightNorm=do_weight_norm,
                             padding=0,
                             **kwargs),
        nn.BatchNorm2d(m * n_band),
        swish(),
    )


def LastBlock(inF, outF, doWeightNorm=False, *args, **kwargs):
    return nn.Sequential(LinearWithConstraint(inF, outF, max_norm=0.5, doWeightNorm=doWeightNorm, *args, **kwargs))



class Classification_FBCNetWithAttn_MultiSpatialConv(nn.Module):

    def __init__(self,
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
                 **kwargs):
        super().__init__()

        # for FBCNet:
        self.m = kwargs.get("m", 32)
        self.n_band = kwargs.get("n_band", 9)  #子频带
        self.n_stride = kwargs.get("n_stride", 4)
        self.n_selayer_reduction = kwargs.get("n_selayer_reduction", 16)
        self.chans = kwargs.get("n_chan", 16)

        self.n_lstm_hidden = kwargs.get("n_lstm_hidden", 64)
        self.n_lstm_layers = kwargs.get("n_lstm_layer", 2)
        self.bidirectional = kwargs.get("bidirectional", True)
        self.lstm_dropout = kwargs.get("lstm_dropout", 0.5)

        # multi spatial conv
        self.spatial_conv_1 = Conv2dWithConstraint(self.n_band,
                                                   8 * self.n_band,
                                                   kernel_size=(3, 1),
                                                   padding="same",
                                                   groups=self.n_band)
        self.spatial_conv_2 = Conv2dWithConstraint(self.n_band,
                                                   8 * self.n_band,
                                                   kernel_size=(5, 1),
                                                   padding="same",
                                                   groups=self.n_band)
        self.spatial_conv_3 = Conv2dWithConstraint(self.n_band,
                                                   8 * self.n_band,
                                                   kernel_size=(3, 1),
                                                   dilation=(2, 1),
                                                   padding="same",
                                                   groups=self.n_band)
        self.spatial_conv_4 = Conv2dWithConstraint(self.n_band,
                                                   8 * self.n_band,
                                                   kernel_size=(5, 1),
                                                   dilation=(2, 1),
                                                   padding="same",
                                                   groups=self.n_band)
        self.batch_norm_1 = nn.BatchNorm2d(8 * self.n_band)
        self.batch_norm_2 = nn.BatchNorm2d(8 * self.n_band)
        self.batch_norm_3 = nn.BatchNorm2d(8 * self.n_band)
        self.batch_norm_4 = nn.BatchNorm2d(8 * self.n_band)

        self.relu = nn.ReLU()

        # 将通道维度进行压缩
        self.scb = nn.Sequential(
            Conv2dWithConstraint(
                32 * self.n_band,
                32 * self.n_band,
                (self.chans, 1),
                #  groups=self.n_band,
                # 这里对每个spatial_conv的输出进行压缩
                groups=self.n_band,
                max_norm=2,
                doWeightNorm=True,
                padding=0,
            ),
            nn.BatchNorm2d(32 * self.n_band),
            swish(),
        )

        self.temporal_layer = LogVarLayer(dim=3)
        self.se_layer = SELayer(channel=self.m * self.n_band, reduction=self.n_selayer_reduction)

        # bi-lstm
        # bi-lstm attention
        self.w = nn.Parameter(torch.zeros(self.n_lstm_hidden * 2))

        self.tanh = nn.Tanh()
        self.rnn = nn.LSTM(input_size=self.m * self.n_band,
                           hidden_size=self.n_lstm_hidden,
                           num_layers=self.n_lstm_layers,
                           batch_first=True,
                           bidirectional=self.bidirectional,
                           dropout=self.lstm_dropout)

        # self.fc = LastBlock(self.m * self.n_band * self.n_stride, num_classes, doWeightNorm=True)
        self.fc = LastBlock(2 * self.n_lstm_hidden, num_classes, doWeightNorm=True)

    def preprocess_input(self, x):
        return x.type(torch.cuda.FloatTensor)

    def forward(self, x,*args):
        x = x.view(x.shape[0], 9, x.shape[2], x.shape[3])
        x = x.to(torch.float32)

        x_1 = self.relu(self.batch_norm_1(self.spatial_conv_1(x))) #torch.Size([64, 72, 19, 512])
        x_2 = self.relu(self.batch_norm_2(self.spatial_conv_2(x)))
        x_3 = self.relu(self.batch_norm_3(self.spatial_conv_3(x)))
        x_4 = self.relu(self.batch_norm_4(self.spatial_conv_4(x)))
        # print('1:',x_1.shape)

        x_1 = x_1.reshape(x_1.shape[0], self.n_band, -1, x_1.shape[2], x_1.shape[3]) #torch.Size([64, 9, 8, 19, 512])
        x_2 = x_2.reshape(x_2.shape[0], self.n_band, -1, x_2.shape[2], x_2.shape[3])
        x_3 = x_3.reshape(x_3.shape[0], self.n_band, -1, x_3.shape[2], x_3.shape[3])
        x_4 = x_4.reshape(x_4.shape[0], self.n_band, -1, x_4.shape[2], x_4.shape[3])
        # print('2:',x_1.shape)

        x = torch.cat([x_1, x_2, x_3, x_4], dim=2) #torch.Size([64, 9, 32, 19, 512])
        # x = torch.cat([x_1, x_2], dim=2)
        # x = torch.cat([x_3, x_4], dim=2)
        # print('3:',x.shape)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4]) #torch.Size([64, 288, 19, 512])
        # print('4:',x.shape)

        x = self.scb(x)  #torch.Size([64, 288, 1, 512])
        # print('5:', x.shape)
        x = x.reshape([*x.shape[0:2], self.n_stride, -1]) #torch.Size([64, 288, 4, 128])
        # print('6:', x.shape)
        band_attentions = self.se_layer(x)
        band_attentions = band_attentions.permute(0, 2, 1).unsqueeze(3)

        temporal_var = self.temporal_layer(x) #torch.Size([64, 288, 4, 1])
        # print('7:', temporal_var.shape)
        # get the attention
        attn_temporal_var = torch.mul(band_attentions, temporal_var)

        var = attn_temporal_var.squeeze(3)#torch.Size([64, 288, 4])
        # print('8:', var.shape)
        var = var.permute(0, 2, 1)
        # print('9:', var.shape) #9: torch.Size([64, 4, 288])
        # # lstm
        H, _ = self.rnn(var)
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        rnn_attn = H * alpha
        rnn_out = torch.sum(rnn_attn, dim=1)
        # print('10:', rnn_out.shape) #torch.Size([64, 128])
        x = self.fc(rnn_out)  #torch.Size([64, 7])
        # print('11:', x.shape)
        x = F.softmax(x, dim=1)

        return x

class FBCNet(nn.Module):

    def __init__(self,
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
                 **kwargs):
        super().__init__()

        # for FBCNet
        self.m = kwargs.get("m", 32)
        self.n_band = kwargs.get("n_band", 9)
        self.n_stride = kwargs.get("n_stride", 4)
        self.n_selayer_reduction = kwargs.get("n_selayer_reduction", 16)

        # model
        self.scb = SCB(self.m, n_chan=chans, n_band=self.n_band, do_weight_norm=True)
        self.temporal_layer = LogVarLayer(dim=3)
        self.fc = LastBlock(self.m * self.n_band * self.n_stride, num_classes, doWeightNorm=True)

    def preprocess_input(self, x):
        return x.type(torch.cuda.FloatTensor)

    def forward(self, x, *args):
        x = x.view(x.shape[0], 9, x.shape[2], x.shape[3])
        print(x.shape)
        x = x.to(torch.float32)

        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.n_stride, -1])

        temporal_var = self.temporal_layer(x)

        var = torch.flatten(temporal_var, start_dim=1)

        # classifier
        x = self.fc(var)
        x = x = F.softmax(x)

        return x

class SharedChannelAttn(nn.Module):
    """
    对齐截图：
    - AvgPool / MaxPool window = 128（这里对最后一维做 pooling，相当于整个窗口）
    - MLP: 288 -> 64 -> 288（共享）
    输入:  x [B, C, K, W]
    输出: att [B, C, K]
    """
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, K, W]
        B, C, K, W = x.shape

        avg_pool = x.mean(dim=-1)              # [B, C, K]
        max_pool = x.max(dim=-1).values        # [B, C, K]

        # 对每个时间窗 K 分别做通道注意力：reshape 成 [B*K, C]
        avg_pool = avg_pool.permute(0, 2, 1).reshape(B * K, C)
        max_pool = max_pool.permute(0, 2, 1).reshape(B * K, C)

        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))

        att = torch.sigmoid(avg_out + max_out)          # [B*K, C]
        att = att.reshape(B, K, C).permute(0, 2, 1)     # [B, C, K]
        return att


class MultiSpatialConv(nn.Module):
    def __init__(self,
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
                 **kwargs):
        super().__init__()

        # ===== 对齐截图的关键超参 =====
        # M=9 子频带
        self.n_band = kwargs.get("n_band", 9)
        # n=8（每个子频带的卷积核数）
        self.n = kwargs.get("n", 8)
        # m=4n=32（4 个分支拼接）
        self.m = kwargs.get("m", 4 * self.n)

        # k=4（时间窗个数），w=128（每个窗长度），这里通过 n_stride=4 + samples=512 保证
        self.n_stride = kwargs.get("n_stride", 4)

        # C=19（通道数）—— 修正你原来写死 16 的问题
        self.chans = int(chans)

        # Att-BiLSTM：layers=2, hidden=64, dropout=0.5
        self.n_lstm_hidden = kwargs.get("n_lstm_hidden", 64)
        self.n_lstm_layers = kwargs.get("n_lstm_layer", 2)
        self.bidirectional = kwargs.get("bidirectional", True)
        self.lstm_dropout = kwargs.get("lstm_dropout", 0.5)

        # ===== multi-branch spatial conv (g=9 分组) =====
        out_ch = self.n * self.n_band  # M*n = 72
        self.spatial_conv_1 = Conv2dWithConstraint(self.n_band, out_ch, kernel_size=(3, 1),
                                                   padding="same", groups=self.n_band)
        self.spatial_conv_2 = Conv2dWithConstraint(self.n_band, out_ch, kernel_size=(5, 1),
                                                   padding="same", groups=self.n_band)
        self.spatial_conv_3 = Conv2dWithConstraint(self.n_band, out_ch, kernel_size=(3, 1),
                                                   dilation=(2, 1), padding="same", groups=self.n_band)
        self.spatial_conv_4 = Conv2dWithConstraint(self.n_band, out_ch, kernel_size=(5, 1),
                                                   dilation=(2, 1), padding="same", groups=self.n_band)

        self.batch_norm_1 = nn.BatchNorm2d(out_ch)
        self.batch_norm_2 = nn.BatchNorm2d(out_ch)
        self.batch_norm_3 = nn.BatchNorm2d(out_ch)
        self.batch_norm_4 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

        # ===== 特征整合与压缩：kernel=19×1, out=M*4n=288, g=9 =====
        c_all = self.m * self.n_band  # 32*9=288
        self.scb = nn.Sequential(
            Conv2dWithConstraint(
                c_all,
                c_all,
                (self.chans, 1),      # 19×1 —— 对齐截图
                groups=self.n_band,   # g=9
                max_norm=2,
                doWeightNorm=True,
                padding=0,
            ),
            nn.BatchNorm2d(c_all),
            swish(),
        )

        # ===== 时域特征 + 共享注意力（288→64→288，avg+max pooling，w=128）=====
        self.temporal_layer = LogVarLayer(dim=3)
        self.shared_attn = SharedChannelAttn(channels=c_all, hidden=64)

        # ===== Att-BiLSTM =====
        self.w = nn.Parameter(torch.zeros(self.n_lstm_hidden * 2))
        self.tanh = nn.Tanh()
        self.rnn = nn.LSTM(
            input_size=c_all,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.lstm_dropout
        )

        # 输出层：128 -> num_classes（截图为 4 类）
        self.fc = LastBlock(2 * self.n_lstm_hidden, num_classes, doWeightNorm=True)

    def forward(self, x, *args):
        # 期望 x: [B, M, C, S] = [B, 9, 19, 512]
        if x.dim() != 4:
            raise ValueError(f"Expected input x with 4 dims [B,M,C,S], got {x.shape}")

        if x.shape[1] != self.n_band:
            # 兼容你原来写死 view 的行为，但用 self.n_band 更稳
            x = x.view(x.shape[0], self.n_band, x.shape[2], x.shape[3])

        x = x.to(torch.float32)

        x_1 = self.relu(self.batch_norm_1(self.spatial_conv_1(x)))  # [B, 72, 19, 512]
        x_2 = self.relu(self.batch_norm_2(self.spatial_conv_2(x)))
        x_3 = self.relu(self.batch_norm_3(self.spatial_conv_3(x)))
        x_4 = self.relu(self.batch_norm_4(self.spatial_conv_4(x)))

        # [B, 72, C, S] -> [B, M, n, C, S]
        x_1 = x_1.reshape(x_1.shape[0], self.n_band, -1, x_1.shape[2], x_1.shape[3])  # [B, 9, 8, 19, 512]
        x_2 = x_2.reshape(x_2.shape[0], self.n_band, -1, x_2.shape[2], x_2.shape[3])
        x_3 = x_3.reshape(x_3.shape[0], self.n_band, -1, x_3.shape[2], x_3.shape[3])
        x_4 = x_4.reshape(x_4.shape[0], self.n_band, -1, x_4.shape[2], x_4.shape[3])

        # 拼 4 分支：n -> 4n = m=32
        x = torch.cat([x_1, x_2, x_3, x_4], dim=2)                  # [B, 9, 32, 19, 512]
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])       # [B, 288, 19, 512]

        x = self.scb(x)                                             # [B, 288, 1, 512]
        x = x.reshape([*x.shape[0:2], self.n_stride, -1])            # [B, 288, 4, 128]  (k=4,w=128)


        temporal_var = self.temporal_layer(x)                        # [B, 288, 4, 1]
        band_attentions = self.shared_attn(x).unsqueeze(3)  # [B, 288, 4, 1]
        attn_temporal_var = band_attentions * temporal_var           # broadcast

        var = attn_temporal_var.squeeze(3)                           # [B, 288, 4]
        var = var.permute(0, 2, 1)                                   # [B, 4, 288]

        H, _ = self.rnn(var)                                         # [B, 4, 2*hidden]
        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        rnn_out = torch.sum(H * alpha, dim=1)                        # [B, 128]

        logits = self.fc(rnn_out)                                    # [B, num_classes]
        return logits  # 重要：返回 logits（不要 softmax），用于 CrossEntropyLoss

if __name__ == '__main__':
    pass
    # print('starting')
    # model = Classification_FBCNetWithAttn_MultiSpatialConv(
    #     log_dir="",
    #     batch_size=32,
    #     epochs=1,
    #     lr=1,
    #     dropout=0.5,
    #     chans=19,
    #     samples=512,
    #     m=4,
    #     n_band=5,
    # )
    # x = torch.randn(32, 5, 19, 512)
    # y = model(x)
    # print(y.shape)
