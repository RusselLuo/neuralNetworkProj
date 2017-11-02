#%%
from dataLoader import LR, train_data, EPOCH
#%%
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch
import os
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # the information per input pixel
                out_channels=16,  # the information per output pixel
                kernel_size=5,  # the filter's size
                stride=1,  # the stride of the filter
                padding=2  # (kernal's size-1)/2
            ),  # The output size is ((W-K+2P)/S)+1
            # W: length of input, K: kernal size, P: padding, S: stride
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # transform the data to feed linear layer
        output = self.out(x)
        return output



if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    net = Net().cuda()
    print(net)
    optimizer = optim.Adam(net.parameters())
    lossFunc = nn.CrossEntropyLoss()
    print('start training\n')
    start = time.time()
    for i in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_data):
            x = Variable(batch_x).cuda()
            # print(x)
            out = net(x)
            loss = lossFunc(out, Variable(batch_y).cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('EPOCH:' + str(i) +
                ' [' + str(step) + ']', ' loss:', str(loss), '\n')
    print('done! used:',time.time()-start)
    # torch.save(net, path+'/trained.pkl')
