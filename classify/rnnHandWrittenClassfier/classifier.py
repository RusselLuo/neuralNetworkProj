
import torch
from dataLoader import train_data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

EPOCH=2
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.001


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64,10)
    def forward(self,x):
        r_out, (h_n,h_c) = self.rnn(x,None) #x->(batch, time_step, input_size)
        out = self.out(r_out[:,-1,:])
        return out
if __name__ == "__main__":
    rnn = RNN().cuda()
    print(rnn)

    optimizer = optim.Adam(rnn.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x,y) in enumerate(train_data):
            x = Variable(x.view(-1,28,28)).cuda()
            y = Variable(y).cuda()
            output = rnn(x)
            loss = loss_func(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%50:
                print(loss)
    torch.save(rnn,"./trained.pkl")