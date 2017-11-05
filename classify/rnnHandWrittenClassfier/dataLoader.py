#%%
import torch.utils.data as Data
import torchvision as vision
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
EPOCH = 5
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD = False

#%%
# path = os.path.dirname(os.path.abspath(__file__))
train_data = vision.datasets.MNIST(
    root='./HandWrittenData',
    train=True,
    transform=vision.transforms.ToTensor(),
    download=DOWNLOAD
)
# plt.imshow(train_data.train_data[1].numpy(),cmap='gray')
# plt.show()
train_data = Data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

