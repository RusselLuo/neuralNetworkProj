#%%
import torch
from torch.autograd import Variable
import pandas as pd
from matplotlib import pyplot as plt
import time
import numpy as np

#%%
iris = pd.DataFrame([l.strip().split(',') for l in open('./iris.data', 'r').readlines()], dtype=int, columns=[
                    'sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
# import data

#%%
net = torch.nn.Sequential(
    torch.nn.Linear(4, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 3)
)


#%%
optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
lossFunc = torch.nn.CrossEntropyLoss()
trainData = iris.sample(frac=0.8, random_state=int(time.time()))
test = iris.drop(trainData.index)

#%%
def getTensor(dataSet):
    tagsDict = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    datas = Variable(torch.Tensor(dataSet.as_matrix()[:, :4].astype(float)))
    tagsMap = [tagsDict[i] for i in dataSet['class'].as_matrix()]
    tags = Variable(torch.LongTensor(tagsMap))
    return datas,tags

#%%
for d in range(10000):
    trainData = iris.sample(frac=0.8, random_state=int(time.time()))
    test = iris.drop(trainData.index)
    datas,tags = getTensor(trainData)
    out = net(datas)
    loss = lossFunc(out, tags)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#test:
datas,tags = getTensor(test)
out = net(datas)
predTags = torch.max(out,1)[1]
same = 0
predTags = predTags.data.numpy()
tags = tags.data.numpy()
for i in range(len(tags)):
    if tags[i]==predTags[i]:
        same+=1
print('predict tags:',predTags)
print('real tags:',tags)
print("accuracy:",same/len(tags))

print('done!!')
