import torch

mat = torch.ones(5,6)
print(mat)
print(torch.cat((mat,mat),0))#增加Y轴
print(torch.cat((mat,mat),1))#增加X轴
print(torch.chunk(mat,2,0))#切割Y轴
print(torch.chunk(mat,2,1))#切割X轴
print(torch.unsqueeze(mat,0))#增加维度到第0维
print(torch.unsqueeze(mat,1))#增加维度到第1维
print(torch.unsqueeze(mat,2))#增加维度到第2维
print(torch.ones(1,2,3))#未压缩的矩阵
print(torch.squeeze(torch.ones(1,2,3)))#将所有的高度为1的维度去除（压缩）