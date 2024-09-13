import numpy as np
import torch as torch
rows=10
cols=12
x=np.arange(1,rows*cols+1).reshape(rows,cols)

print(x)

fold = cols // 3
fold_1_12 = cols // 12
out = np.zeros((rows, cols))
out[:-1, :fold] = x[1:, :fold]  # shift left
print (2 * x[-2, :fold])
print (x[-3, :fold])
out[-1, :fold] = (2 * x[-1, :fold] - x[-2, :fold]) / 1  # left差分
out[1:, fold: 2 * fold] = x[:-1, fold: 2 * fold]  # shift right
out[0, fold: 2 * fold] = (2 * x[0, fold: 2 * fold] - x[1, fold: 2 * fold]) / 1  # right差分
out[:, 2 * fold:] = x[:, 2 * fold:]  # not shift
out[:-1, fold - fold_1_12: fold] = out[1:, fold - fold_1_12: fold]  # 前1/3的1/12 left
out[-1, fold - fold_1_12: fold] = 0  # 补0

out[1:, 2*fold-fold_1_12: 2*fold] = out[:-1,  2*fold-fold_1_12: 2*fold]  # 中间1/3的1/12 right
out[0, 2*fold-fold_1_12: 2*fold] = 0 # 补0
print (out)
