import torch
from torch.autograd import Variable
import torchvision as vision
from classifier import Net


net = torch.load('./trained.pkl')


test_data = vision.datasets.MNIST(
    root='./HandWrittenData',
    train=False,
    transform=vision.transforms.ToTensor(),
    download=False
)
# the "unsqueeze" method is transforming 10000*0 1-D data to 10000*1 2-D data
test_x = (Variable(torch.unsqueeze(test_data.test_data, 1).type(
    torch.FloatTensor), volatile=True) / 255.).cuda()
# volatile is optimizing the memory using when you are sure that you won't call backward
test_y = test_data.test_labels.numpy()

predict = net(test_x).cpu().data
predict = torch.max(predict, 1)[1].numpy()
print(test_y)
print(predict)
same = 0
for i in range(len(test_y)):
    if test_y[i] == predict[i]:
        same += 1
print('accuracy:', same / len(test_y))
# predict = net()
