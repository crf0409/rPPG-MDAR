"""Temporal Shift Convolutional Attention Network (TS-CAN).
Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
NeurIPS, 2020
Xin Liu, Josh Fromm, Shwetak Patel, Daniel McDuff
"""
import time

import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


# 1/3差分，1/12补0

# class TSM(nn.Module):
#     def __init__(self, n_segment=10, fold_div=3):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         # fold = c // self.fold_div
#         fold = c // 4
#         fold_1_12 = c // 12
#         out = torch.zeros_like(x)
#         # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         # out[:, -1, :fold] = (1.8 * x[:, -3, :fold] - 0.2 * x[:, -1, :fold] -0.6 *  x[:, -2, :fold]) / 4 # left差分 1.8c-0.2a-0.6b
#         # out[:, -1, :fold] = (2 * x[:, -1, :fold] - x[:, -2, :fold])/1 # left差分
#         # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#         # out[:, 0, fold: 2 * fold] = (1.8 * x[:, 2, fold: 2 * fold] - 0.2 * x[:, 0, fold: 2 * fold] - 0.6 * x[:, 1, fold: 2 * fold]) / 4 # right差分 1.8c-0.2a-0.6b
#         # out[:, 0, fold: 2 * fold] = (2 * x[:, 0, fold: 2 * fold] - x[:, 1, fold: 2 * fold])/1  # right差分
#         # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#         #out[:, :-1, fold - fold_1_12: fold] = out[:, 1:, fold - fold_1_12: fold]  # 前1/3的1/12 left
#         # out[:, -1, fold - fold_1_12: fold] = 0  # 补0
#         #out[:, 1:, 2 * fold - fold_1_12: 2 * fold] = out[:, :-1, 2 * fold - fold_1_12: 2 * fold]  # 中间1/3的1/12 right
#         # out[:, 0, 2 * fold - fold_1_12: 2 * fold] = 0  # 补0
#         # print(out.view(nt, c, h, w).shape)
#
#         # 1/4 左
#         out[:, 1:, :fold] = x[:, :-1, :fold]
#         out[:, 0, :fold] = 2 * x[:, 0, :fold] - x[:, 1, :fold]
#
#         # 1/4 右
#         out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
#         out[:, -1, fold:2 * fold] = 2 * x[:, -1, fold:2 * fold] - x[:, -2, fold:2 * fold]
#
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#
#         return out.view(nt, c, h, w)


# 1/3差分，1/12补0end

#TSM差分

# class TSM(nn.Module):
#     def __init__(self, n_segment=10, fold_div=3):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         fold = c // self.fold_div
#         out = torch.zeros_like(x)
#         out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         out[:, -1, :fold] = 2 * x[:, -2, :fold] - x[:, -3, :fold]  #推算
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#         out[:, 0, fold: 2 * fold] = 2 * x[:, 1, fold: 2 * fold] - x[:, 2, fold: 2 * fold]  #推算
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#         # print(out.view(nt, c, h, w).shape)
#         return out.view(nt, c, h, w)

#TSM差分end


# 原本的
# class TSM(nn.Module):
#     def __init__(self, n_segment=10, fold_div=3):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         fold = c // self.fold_div
#         out = torch.zeros_like(x)
#         out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#         return out.view(nt, c, h, w)

# 原本的end


# class TSM(nn.Module):
#     def __init__(self, n_segment=10):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         fold = c // 3
#         out = torch.zeros_like(x)
#
#         # shift left
#         out[:, :-1, :fold] = x[:, 1:, :fold]
#         out[:, -1, :fold] = x[:, 0, :fold]
#
#         # shift right
#         out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
#         out[:, 0, fold:2 * fold] = x[:, -1, fold:2 * fold]
#
#         # not shift
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
#
#         #融
#         out = (out[:, :, :fold] + out[:, :, fold:2 * fold] + out[:, :, 2 * fold:]) / 3
#         print(out.shape)
#         return out.view(nt, c, h, w)
#
#
# class TSM(nn.Module):
#     def __init__(self, n_segment=10, fold_div=3):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         fold = c // self.fold_div
#         out = torch.zeros_like(x)
#
#         out[:, :-1, :fold] = x[:, 1:, :fold]
#         out[:, -1, :fold] = x[:, 0, :fold]
#
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
#         out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]
#
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#
#         return out.view(nt, c, h, w)

# 1/12偏移
# class TSM(nn.Module):
#     def __init__(self, n_segment=10):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#
#     def forward(self, x):
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
#         out[:, 0, :fold_1_4] = x[:, -1, :fold_1_4]
#
#         # 1/4 右
#         out[:, :-1, fold_1_4:2 * fold_1_4] = x[:, 1:, fold_1_4:2 * fold_1_4]
#         out[:, -1, fold_1_4:2 * fold_1_4] = x[:, 0, fold_1_4:2 * fold_1_4]
#
#         # 1/12 左
#         out[:, 2:, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, :-2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]
#         out[:, 0,
#         2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, -2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]
#         out[:, 1, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, -1, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]
#
#         # 1/12 右
#         out[:, :-2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, 2:,
#                                                                              2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12]
#         out[:, -2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, 0,
#                                                                             2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12]
#         out[:, -1, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, 1,
#                                                                             2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12]
#
#         # 1/3 不动
#         out[:, :, 2 * fold_1_4 + 2 * fold_1_12:] = x[:, :, 2 * fold_1_4 + 2 * fold_1_12:]
#
#         return out.view(nt, c, h, w)
# 1/12偏移end


# 光流法
# def calculate_optical_flow(prev_frame, curr_frame):
#     # 计算前一帧和当前帧之间的光流
#     # 这里使用简单的差分法计算光流,实际应用中可以使用更advanced的光流算法
#     flow = curr_frame - prev_frame
#     return flow
#
#
# def warp_frame(prev_frame, flow):
#     # 使用光流对前一帧进行变形,生成当前帧
#     # 这里使用简单的光流加法进行变形,实际应用中可以使用更advanced的变形算法
#     warped_frame = prev_frame + flow
#     return warped_frame
# #
#
# class TSM(nn.Module):
#     def __init__(self, n_segment=10):
#         super(TSM, self).__init__()
#         self.n_segment = n_segment
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#
#         # 使用光流法生成新的特征向量
#         new_frames = []
#         for i in range(self.n_segment - 1):
#             prev_frame = x[:, i]
#             curr_frame = x[:, i + 1]
#             flow = calculate_optical_flow(prev_frame, curr_frame)
#             new_frame = warp_frame(prev_frame, flow)
#             new_frame = new_frame.unsqueeze(1)  # 在第二个维度上增加一个维度
#             new_frames.append(new_frame)
#
#         # 将新生成的特征向量与原始的第一帧拼接
#         out = torch.cat([x[:, 0:1], *new_frames], dim=1)
#
#         # 舍弃多余的帧
#         out = out[:, :self.n_segment]
#
#         return out.view(nt, c, h, w)
#光流法end

#仅针对1/12进行差分帧
#


# 仅针对1/12进行差分帧end
#
# 1/4缺失部分差分方法补齐，1/12缺失部分第一帧用差分法补齐，第二针补0  jiaohuantongdao

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
#         out[:, 0, :fold_1_4] = (2*x[:, 1, :fold_1_4]-x[:, 2, :fold_1_4])/4#1,2
#
#         # 1/4 右
#         out[:, :-1, fold_1_4:2 * fold_1_4] = x[:, 1:, fold_1_4:2 * fold_1_4]
#         out[:, -1, fold_1_4:2 * fold_1_4] = (
#                 (2*x[:, -2, fold_1_4:2 * fold_1_4]-x[:, -3, fold_1_4:2 * fold_1_4])/4)#-1,-2
#
#         # 1/12 左
#         out[:, 2:, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, :-2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]
#         out[:, 1, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = (
#                 (2*x[:, 2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]-x[:, 3, 2 * fold_1_4:2 * fold_1_4 + fold_1_12])/4)#1,2
#         # 1/12 右
#         out[:, :-2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, 2: , 2 * fold_1_4 + fold_1_12 : 2 * fold_1_4 + 2 * fold_1_12]
#         out[:, -2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = (
#                 (2*x[:, -3, 2 * fold_1_4 + fold_1_12 : 2 * fold_1_4 + 2 * fold_1_12]-x[:, -4, 2 * fold_1_4 + fold_1_12 : 2 * fold_1_4 + 2 * fold_1_12])/4)
#         #-1,-2
#         # 1/3 不动
#         out[:, :, 2 * fold_1_4 + 2 * fold_1_12:] = x[:, :, 2 * fold_1_4 + 2 * fold_1_12:]
#
#         return out.view(nt, c, h, w)

#
# # # 1/4缺失部分差分方法补齐，1/12缺失部分第一帧用差分法补齐，第二针补0 end

class TSM(nn.Module):
    def __init__(self, n_segment=10):
        super(TSM, self).__init__()
        self.n_segment = n_segment

    def forward(self, x):

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold_1_4 = c // 4
        fold_1_12 = c // 12
        fold_1_3 = c // 3

        out = torch.zeros_like(x)

        # 1/4 左
        out[:, 1:, :fold_1_4] = x[:, :-1, :fold_1_4]
        out[:, 0, :fold_1_4] = x[:, 0, :fold_1_4]/8

        # 1/4 右
        out[:, :-1, fold_1_4:2 * fold_1_4] = x[:, 1:, fold_1_4:2 * fold_1_4]
        out[:, -1, fold_1_4:2 * fold_1_4] = x[:, -1, fold_1_4:2 * fold_1_4]/8

        # 1/12 左
        out[:, 2:, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, :-2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]
        out[:, 1, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, 0, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]/8
        # 1/12 右
        out[:, :-2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, 2: , 2 * fold_1_4 + fold_1_12 : 2 * fold_1_4 + 2 * fold_1_12]
        out[:, -2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, -1, 2 * fold_1_4 + fold_1_12 : 2 * fold_1_4 + 2 * fold_1_12]/8
        # 1/3 不动
        out[:, :, 2 * fold_1_4 + 2 * fold_1_12:] = x[:, :, 2 * fold_1_4 + 2 * fold_1_12:]

        return out.view(nt, c, h, w)

#取diff平均值填充
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
#         fold_1_4 = c // 3
#         fold_1_12 = c // 12
#         fold_1_3 = c // 3
#
#         out = torch.zeros_like(x)
#
#         absmeanStart =torch.sign(x)*torch.mean(torch.abs(x)[:,0:4], dim=1,keepdim=True)#所有channel列的绝对值平均值
#         absmeanEnd = torch.sign(x)[:, -1,:] * torch.mean(torch.abs(x)[:,-1:4], dim=2)[:, :]
#         # print(torch.sign(x)[:,0,:].size())
#         # print(torch.mean(torch.abs(x), dim=1).size())
#         # print (absmean.size())
#         # print(absmean[0,0,:,0,0])
#         # exit()
#         # 1/4 左
#         out[:, 1:, :fold_1_4] = x[:, :-1, :fold_1_4]
#         # out[:, 0, :fold_1_4] = x[:, 0, :fold_1_4]
#         # print(1111,out[0, 0, 0])
#         # print(3333, out[0, 1, 0])
#         out[:, 0, :fold_1_4]=absmean[:, 0, :fold_1_4]/8
#         # print(2222,out[0, 0, 0])
#
#         # 1/4 右
#         out[:, :-1, fold_1_4:2 * fold_1_4] = x[:, 1:, fold_1_4:2 * fold_1_4]
#         # out[:, -1, fold_1_4:2 * fold_1_4] = (2 * x[:, -2, fold_1_4:2 * fold_1_4] - x[:, -3, fold_1_4:2 * fold_1_4])/8
#         out[:, -1, fold_1_4:2 * fold_1_4] = absmean[:, -1, fold_1_4:2 * fold_1_4]/8
#         # # 1/12 左
#         # out[:, 2:, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = x[:, :-2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12]
#         # # 通过前后帧的像素趋势推理缺失的两帧
#         # out[:, 1, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] = (2 * x[:, 2, 2 * fold_1_4:2 * fold_1_4 + fold_1_12] - x[:, 3,2 * fold_1_4:2 * fold_1_4 + fold_1_12])/8
#         # # 1/12 右
#         # out[:, :-2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = x[:, 2: , 2 * fold_1_4 + fold_1_12 : 2 * fold_1_4 + 2 * fold_1_12]
#         # # 通过前后帧的像素趋势推理缺失的两帧
#         # out[:, -2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] = (2 * x[:, -3, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] - x[:,-4,2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12])/8
#         # # out[:, -2, 2 * fold_1_4 + fold_1_12:2 * fold_1_4 + 2 * fold_1_12] =(2 * x[:, -3,2 * fold_1_4:2 * fold_1_4 + fold_1_12] - x[:,-4,2 * fold_1_4:2 * fold_1_4 + fold_1_12])/8
#         # # 1/3 不动
#         # out[:, :, 2 * fold_1_4 + 2 * fold_1_12:] = x[:, :, 2 * fold_1_4 + 2 * fold_1_12:]
#         out[:, :, 2 * fold_1_4:] = x[:, :, 2 * fold_1_4:]
#
#         #print(out.view(nt, c, h, w).shape)
#         return out.view(nt, c, h, w)


# # 取diff平均值填充

# class TSM1(nn.Module):#yuanban
#     def __init__(self, n_segment=10, fold_div=3):
#         super(TSM1, self).__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         fold = c // self.fold_div
#         out = torch.zeros_like(x)
#         out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#         return out.view(nt, c, h, w)
#
# class TSM2(nn.Module):#linjinzhen buwei
#     def __init__(self, n_segment=10, fold_div=3):
#         super(TSM2, self).__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#
#     def forward(self, x):
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         fold = c // self.fold_div
#         out = torch.zeros_like(x)
#         out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
#         out[:, -1, :fold] = x[:, -1, :fold]/8
#         out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
#         out[:, 0, fold: 2 * fold] = x[:, 0, fold: 2 * fold]/8
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#         return out.view(nt, c, h, w)
#
# class TSM3(nn.Module):
#     def __init__(self, n_segment=10, fold_div=3):
#         super(TSM3, self).__init__()
#         self.n_segment = n_segment
#         self.fold_div = fold_div
#
#     def forward(self, x):
#
#         nt, c, h, w = x.size()
#         n_batch = nt // self.n_segment
#         x = x.view(n_batch, self.n_segment, c, h, w)
#         fold = c // self.fold_div
#         out = torch.zeros_like(x)
#
#         out[:, :-2, :fold] = x[:, 2:, :fold]  # shift left
#         out[:, -2, :fold] = x[:, -1, :fold]/8
#         out[:, 2:, fold: 2 * fold] = x[:, :-2, fold: 2 * fold]  # shift right
#         out[:, 1, fold: 2 * fold] = x[:, 0, fold: 2 * fold]/8
#         out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
#
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
        self.dropout_4 = nn.Dropout(self.dropout_rate2)
        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        elif img_size == 128:
            self.final_dense_1 = nn.Linear(57600, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')
        self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def forward(self, inputs, params=None):
        time_start = time.time()
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        diff_input = self.TSM_1(diff_input)

        # print(bb-aa,"111")
        d1 = torch.tanh(self.motion_conv1(diff_input))

        d1 = self.TSM_2(d1)

        # print(bb - aa, "222")
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

        # print("222",d4.size(),g1.size(),gated1.size(),r4.size(),r3.size(),r2.size())
        # exit()
        d5 = torch.tanh(self.motion_conv3(d4))

        d5 = self.TSM_4(d5)

        # print(bb - aa, "333")

        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)
        time_end = time.time()
        time_c = time_end - time_start
        # print(time_c)

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
