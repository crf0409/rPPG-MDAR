"""Temporal Shift Convolutional Attention Network (TS-CAN).
Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
NeurIPS, 2020
Xin Liu, Josh Fromm, Shwetak Patel, Daniel McDuff
"""
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class STNNet(nn.Module):
    def __init__(self):
        super(STNNet, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=0.25)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(25088, 256),  # 调整输入维度为 25088
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(x.size(0), -1)  # 使用 x.size(0) 作为批次大小
        # print(xs.size())
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = F.upsample(x, size=(70, 70), mode='bilinear', align_corners=False)
        # print(x.size())
        return x


# class Attention_mask(nn.Module):
#     def __init__(self):
#         super(Attention_mask, self).__init__()
#
#     def forward(self, x):
#         x = F.softmax(x, dim=-2)
#         x = F.softmax(x, dim=-1)
#         return x

class SpatialAttention(nn.Module):
    def __init__(self, inc):
        super(SpatialAttention, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=inc, out_channels=1, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=inc, out_channels=1, kernel_size=3, padding=1, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=inc, out_channels=1, kernel_size=5, padding=2, bias=True)
        # self.pool = nn.MaxPool2d(3, stride=1,padding=1)
        # self.conv4 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0, bias=True)
        # self.sigmoid = nn.Sigmoid()
        # self.batchnorm1 = nn.BatchNorm2d(1)
        # self.batchnorm2 = nn.BatchNorm2d(1)
        # self.batchnorm3 = nn.BatchNorm2d(1)
        # self.batchnorm4 = nn.BatchNorm2d(1)
        # self.scale = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5，但作为可学习的参数

    def forward(self, x):

        x1 = self.conv2(x)
        # x1 = self.batchnorm1(x1)
        x1 = F.silu(x1)
        x = x1
        # print(x)
        # print(x.mean())
        # exit()

        # x2 = self.conv11(x1)
        # x2 = self.batchnorm2(x2)
        # x2 = F.relu(x2)
        #
        # x3 = self.conv11(x2)
        # x3 = self.batchnorm2(x3)
        # x3 = F.relu(x3)
        #
        # x = x1+x2+x3
        # # x = self.batchnorm3(x)
        # x = F.relu(x)

        # x = self.batchnorm(x)
        # x = self.sigmoid(x)
        # 假设x的形状为 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # 将第三维（height）和第四维（width）展开
        # print(x)
        x = x.view(batch_size, channels, -1)  # 变成 (batch_size, channels, height * width)
        # # 计算特征平均值，并进行缩放，使得平均值为1
        mean_val = x.mean()    #   计算特征平均值，形状为 (batch_size, channels, 1)


        # print(mean_val)
        # exit()
        # scale_factor = 0.5  # 根据标准差动态调整
        # print(x.std())
        x = x / mean_val * 0.2 # 缩放特征，使得平均值为1
        # print(x.std())
        # exit()
        # 重新将第二维展开为第三维和第四维
        x = x.view(batch_size, channels, height, width)

        # print(x)
        # print(x.size())
        # exit()
        # x_min = torch.min(x)
        # x_max = torch.max(x)
        # x = (x - x_min) / (x_max - x_min)/2+0.75
        return x
#
class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        # print(x / xsum * xshape[2] * xshape[3] * 0.5)
        # exit()
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config



# 定义ECANet的类
class eca_block(nn.Module):
    # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
    def __init__(self, in_channel=72, b=1, gama=2):
        # 继承父类初始化
        super(eca_block, self).__init__()

        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
        # 如果卷积核大小是奇数，就使用它
        if kernel_size % 2:
            kernel_size = kernel_size
        # 如果卷积核大小是偶数，就把它变成奇数
        else:
            kernel_size = kernel_size

        # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
        padding = kernel_size // 2

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              bias=False, padding=padding)
        # sigmoid激活函数，权值归一化
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获得输入图像的shape
        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs

# 原版TSM
class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)

###测试TSM集合
# class TSM(nn.Module):
#     def __init__(self, n_segment=10):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#
#     def forward(self, x):
#
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#
#         fold_1_4 = c // 4
#         fold_1_12 = c // 12
#         fold_1_6 = c // 6
#         fold_1_8 = c // 8
#         fold_1_3 = c // 3
#         fold_1_18 = c // 18
#
#         out = torch.zeros_like(x)
#
#         #
#         out[:, 1:, : fold_1_6] = x[:, :-1, : fold_1_6]
#         #
#         out[:, :-1,  fold_1_6:2 * fold_1_6] = x[:, 1:,fold_1_6:2 * fold_1_6]
#         #
#         out[:, 1:, 2 * fold_1_6:2 * fold_1_6+fold_1_6] = x[:, :-1,  2 * fold_1_6:2 * fold_1_6+fold_1_6]
#
#         out[:, :-1,  2 * fold_1_6+fold_1_6: 2 * fold_1_6+2*fold_1_6 ] = x[:, 1: , 2 * fold_1_6+fold_1_6: 2 * fold_1_6+2*fold_1_6 ]
#
#         out[:, :, 4 * fold_1_6:] = x[:, :, 4  * fold_1_6:]
#         return out.view(nt, c, h, w)

###1/4左1/4右1/12左1/12右剩下的不动
# class TSM(nn.Module):
#     def __init__(self, n_segment=10):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#
#     def forward(self, x):
#
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#
#         fold_1_4 = c // 4
#         fold_1_12 = c // 12
#         fold_1_3 = c // 3
#
#         out = torch.zeros_like(x)
#
#         # 1/4 左
#         out[:, 1:, :fold_1_4] = x[:, :-1, :fold_1_4]
#         # 1/4 右
#         out[:, :-1, fold_1_4:2 * fold_1_4] = x[:, 1:, fold_1_4:2 * fold_1_4]
#         # 1/12 左
#         out[:, 2:, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, :-2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]
#         # 1/12 右
#         out[:, :-2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, 2: , 2 * fold_1_4 + fold_1_12 : 2 * fold_1_4 + 2 * fold_1_12]
#         # 1/3 不动
#         out[:, :, 2 * fold_1_4 + 2 * fold_1_12:] = x[:, :, 2 * fold_1_4 + 2 * fold_1_12:]
#         # print(out.view(nt,c,h,w).size())
#         return out.view(nt, c, h, w)


class TSCAN(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20, img_size=36):
        """Definition of TS_CAN.
        Args:
          in_channels: the number of input channel. Default: 3
          frame_depth: the number of frame (window size) used in temport shift. Default: 20
          img_size: height/width of each frame. Default: 36.
        Returns:
          TS_CAN model.
        """
        super(TSCAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # self.stn_net = STNNet()
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_4 = nn.AvgPool2d(self.pool_size)

        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
            # self.final_dense_1 = nn.Linear(12288, self.nb_dense, bias=True)
            # self.final_dense_1 = nn.Linear(40960, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(57600, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)
        # self.reduce_conv1 = nn.Conv2d(160, 48, kernel_size=3, padding=(1, 1),bias=True)
        self.spatialAttention1 = SpatialAttention(inc=self.nb_filters1)
        self.spatialAttention2 = SpatialAttention(inc=self.nb_filters2)
        self.batchnorm1 = nn.BatchNorm2d(self.nb_filters1)
        self.batchnorm2 = nn.BatchNorm2d(self.nb_filters2)
        self.avg_pooling_5 = nn.AvgPool2d(self.pool_size)
        # self.c2f1 = C2f(self.nb_filters1,self.nb_filters1)
        # self.c2f2 = C2f(self.nb_filters2,self.nb_filters2)

    def forward(self, inputs, params=None):
        # print(111,self.nb_dense)
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]
        # print(f"diff{diff_input.size()}")
        # print(f"raw{raw_input.size()}")
        # np.save('diff_input.npy',diff_input).cpu()


        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        # print(11, d1.size())
        d2 = torch.tanh(self.motion_conv2(d1))
        # print(12, d2.size())

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))
        # r2 = self.stn_net(r1)
        # print(11111, r1.size())
        # print(2, r2.size())
        # exit()
        # r2=self.c2f1(r2)
        # g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        # print(g1)
        # exit()
        # g1 = self.attn_mask_1(g1)
        # g1 = self.c2f1(r2)
        g1 = self.spatialAttention1(r2)
        # print(torch.mean(g1))
        # print(torch.mean(d2))
        # exit()
        # print(d2.std(), d2.mean())
        gated1 = d2 * g1
        # print(gated1.std(), gated1.mean())
        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)
        # print(13, d3.size())


        # d22 = self.apperance_att_conv3(d2)
        # d22 = self.attn_mask_3(d22)
        # d22 = self.spatialAttention1(d2)
        # print(d2)
        # print(d22.std(),d22.mean())
        # print(r2.std(),r2.mean())
        # exit()
        # r2 = r2*(d22/5)
        # print(r2.std(), r2.mean())
        # exit()
        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)
        # print(3, r3.size())
        # print(4, r4.size())

        d4 = self.TSM_3(d4)
        # print(14, d4.size())
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        # print(15, d5.size())
        d6 = torch.tanh(self.motion_conv4(d5))
        # print(16, d6.size())

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))
        # print(5, r5.size())
        # print(6, r6.size())
        # r6=self.c2f2(r6)
        # g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        # g2 = self.attn_mask_2(g2)
        # g2 = self.c2f2(r6)
        g2 = self.spatialAttention2(r6)
        # print(g2)
        # print(d6.std(), d6.mean())
        gated2 = d6 * g2
        # print(gated2.std(),gated2.mean())
        # exit()

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        # p2 = F.interpolate(d2, size=(16, 16), mode='bilinear', align_corners=False)
        # d8 = torch.cat((d8, self.avg_pooling_5(d6), p2), dim=1)
        # d8 = self.reduce_conv1(d8)
        # print(17, d7.size())
        # print(18, d8.size())
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        # print(19, d9.size())
        # print(110, d10.size())
        # print(111, d11.size())

        out = self.final_dense_2(d11)

        return out


class MTTS_CAN(nn.Module):
    """MTTS_CAN is the multi-task (respiration) version of TS-CAN"""

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, dropout_rate1=0.25,
                 dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128, frame_depth=20):
        super(MTTS_CAN, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        # TSM layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)
        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1),
                                      bias=True)
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Apperance branch convs
        self.apperance_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, bias=True)
        self.apperance_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size,
                                         padding=(1, 1), bias=True)
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, bias=True)
        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(
            self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(
            self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()
        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)
        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4_y = nn.Dropout(self.dropout_rate2)
        self.dropout_4_r = nn.Dropout(self.dropout_rate2)

        # Dense layers
        self.final_dense_1_y = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_y = nn.Linear(self.nb_dense, 1, bias=True)
        self.final_dense_1_r = nn.Linear(16384, self.nb_dense, bias=True)
        self.final_dense_2_r = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        diff_input = self.TSM_1(diff_input)
        d1 = torch.tanh(self.motion_conv1(diff_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)

        d10 = torch.tanh(self.final_dense_1_y(d9))
        d11 = self.dropout_4_y(d10)
        out_y = self.final_dense_2_y(d11)

        d10 = torch.tanh(self.final_dense_1_r(d9))
        d11 = self.dropout_4_r(d10)
        out_r = self.final_dense_2_r(d11)

        return out_y, out_r